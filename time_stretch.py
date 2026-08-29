from typing import Sequence

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.window import Window

from time_periods import (
    Aggregation,
    PeriodPartition,
    build_agg_expressions,
    null_safe_join,
    split_on_period_borders,
    validate_aggs,
    validate_columns,
    validate_period_column,
    validate_period_partition,
    validate_periods,
)

# Internal helper columns. Prefixed so they cannot collide with user columns.
_PREV_MAX_END = "_ts_prev_max_end"
_IS_NEW_CHAIN = "_ts_is_new_chain"
_LOCAL_CHAIN = "_ts_local_chain"
_CHAIN = "_ts_chain"
_LOCAL_START = "_ts_local_start"
_LOCAL_END = "_ts_local_end"


def _validate(
        df: DataFrame,
        group_by_columns: Sequence[str],
        aggs: Sequence[Aggregation],
        max_interval_seconds: int,
        start_time_column: str,
        end_time_column: str,
        period,
) -> list:
    """Check every referenced column exists and every agg spec is usable.

    Returns the list of aggregation expressions to apply to each stretched
    segment. Raises ValueError on the first problem found, reporting all
    missing columns at once.
    """
    if not group_by_columns:
        raise ValueError("group_by_columns must not be empty")
    if max_interval_seconds < 0:
        raise ValueError(f"max_interval_seconds must be >= 0, got {max_interval_seconds}")

    agg_columns = validate_aggs(aggs)
    validate_columns(df, [*group_by_columns, start_time_column, end_time_column, *agg_columns])

    if period is not None:
        # The three-step plan below only pays off while a chain can bridge at
        # most a couple of periods; a gap longer than the period itself means
        # partition-local stretching merges nothing.
        validate_period_partition(period, max_interval_seconds,
                                  f"to partition by {PeriodPartition.COLUMN!r}")
        validate_period_column(df, period)

    # The stretched segment spans its members; the user aggs come after it.
    span = [
        F.min(F.col(start_time_column)).alias(start_time_column),
        F.max(F.col(end_time_column)).alias(end_time_column),
    ]
    reserved = {start_time_column, end_time_column, *group_by_columns}
    return span + build_agg_expressions(aggs, reserved)


def _assign_chain_ids(
        df: DataFrame,
        partition_columns: Sequence[str],
        start_time_column: str,
        end_time_column: str,
        max_interval_seconds: int,
        out_column: str,
) -> DataFrame:
    """Number the stretched chains within each window partition.

    Sorted by start, a row opens a new chain when the gap between its start and
    the largest end seen so far exceeds max_interval_seconds (a negative gap
    means the rows overlap). A running sum of those "new chain" flags gives a
    chain id that is unique within the partition.
    """
    ordered = (
        Window.partitionBy(*partition_columns)
        .orderBy(F.col(start_time_column).asc(), F.col(end_time_column).asc())
    )
    preceding = ordered.rowsBetween(Window.unboundedPreceding, -1)
    running = ordered.rowsBetween(Window.unboundedPreceding, Window.currentRow)

    # cast("long") turns a timestamp into epoch seconds, and leaves an
    # already-numeric (epoch seconds) column alone.
    gap = F.col(start_time_column).cast("long") - F.col(_PREV_MAX_END).cast("long")

    return (
        df.withColumn(_PREV_MAX_END, F.max(F.col(end_time_column)).over(preceding))
        .withColumn(_IS_NEW_CHAIN, F.col(_PREV_MAX_END).isNull() | (gap > F.lit(max_interval_seconds)))
        .withColumn(out_column, F.sum(F.col(_IS_NEW_CHAIN).cast("long")).over(running))
        .drop(_PREV_MAX_END, _IS_NEW_CHAIN)
    )


def time_stretch(
        df: DataFrame,
        group_by_columns: list[str],
        aggs: list[Aggregation],
        max_interval_seconds: int,
        start_time_column: str,
        end_time_column: str,
        period: PeriodPartition = None,
) -> DataFrame:
    """Merge time segments that overlap, or sit no more than
    max_interval_seconds apart, into a single stretched segment.

    Rows are grouped by `group_by_columns`; within a group the segments are
    walked in start order and each chain of near/overlapping segments collapses
    to one output row spanning [min(start), max(end)], with `aggs` applied over
    the chain's members.

    Args:
    - df: the segments.
    - group_by_columns: the columns identifying an independent timeline.
    - aggs: (Spark aggregation function, [column names]) pairs -- F.max,
      F.min, F.sum, F.avg, F.count, ... -- e.g.
      [(F.sum, ["bytes"]), (F.max, ["bytes", "score"])]. Each pair yields one
      output column per input column, named "{column}_{function}"
      (bytes_sum, bytes_max, score_max).
    - max_interval_seconds: segments this far apart (or closer) are stretched
      together; 0 merges only overlapping/touching segments.
    - start_time_column, end_time_column: the segment bounds, timestamps or
      epoch seconds.
    - period: optional partitioning of the data. Every input segment must lie
      inside one period (validated up front). The work is then done in three
      steps: stretch inside each (group, period) partition, so the expensive
      per-group sort runs on partition-sized data; stitch the partial chains
      that continue across a border; split the stretched segments back on the
      period borders, so the output stays partitioned. A chain that crossed a
      border therefore comes back as one row per period, each carrying the
      whole chain's aggregated values.

    Returns:
    - DataFrame: group_by_columns, the stretched start_time_column and
      end_time_column, then one column per aggregation (plus period.column
      when a period is given).

    Raises:
    - ValueError: a referenced column is missing, an aggs entry is malformed,
      the period's partition column is absent / shorter than
      max_interval_seconds, or a segment is not inside a single period.
    - AnalysisException: an aggs function is not an aggregation (Spark's
      analyzer rejects it while the plan is built, inside this call).
    """
    agg_exprs = _validate(
        df, group_by_columns, aggs, max_interval_seconds,
        start_time_column, end_time_column, period,
    )

    if period is None:
        chained = _assign_chain_ids(
            df, group_by_columns, start_time_column, end_time_column,
            max_interval_seconds, _CHAIN,
        )
    else:
        validate_periods(df, start_time_column, end_time_column, period)

        # Step 1: stretch inside each (group, period) partition. Nothing here
        # sees across a partition border yet.
        local = _assign_chain_ids(
            df, [*group_by_columns, period.column], start_time_column, end_time_column,
            max_interval_seconds, _LOCAL_CHAIN,
        )
        # Step 2: each local chain becomes a single interval, and those are
        # stretched again per group, merging the chains that continue across a
        # border. Chain merging is associative -- a local chain spans exactly
        # the region its members cover, and a longer end can only shrink the
        # gap to the next segment -- so stitching the partial chains
        # reproduces the global chains.
        local_spans = (
            local.groupBy(*group_by_columns, period.column, _LOCAL_CHAIN)
            .agg(F.min(F.col(start_time_column)).alias(_LOCAL_START),
                 F.max(F.col(end_time_column)).alias(_LOCAL_END))
        )
        stitched = _assign_chain_ids(
            local_spans, group_by_columns, _LOCAL_START, _LOCAL_END,
            max_interval_seconds, _CHAIN,
        ).select(*group_by_columns, period.column, _LOCAL_CHAIN, _CHAIN)

        # Carry the global chain id back to the original rows, so the user
        # aggregations run once over real rows rather than over partial results
        # (which would break for non-decomposable aggregations such as avg).
        join_keys = [*group_by_columns, period.column, _LOCAL_CHAIN]
        chained = null_safe_join(local, stitched, join_keys).select("l.*", f"r.`{_CHAIN}`")

    stretched = (
        chained.groupBy(*group_by_columns, _CHAIN)
        .agg(*agg_exprs)
        .drop(_CHAIN)
    )
    if period is None:
        return stretched
    # Step 3: a stitched chain may now cross a border, so cut it back up.
    return split_on_period_borders(stretched, start_time_column, end_time_column, period)
