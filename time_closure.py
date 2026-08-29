import pyspark.sql.functions as F
from pyspark.sql import Column, DataFrame

from time_periods import (
    PERIOD_INDEX,
    PeriodPartition,
    null_safe_condition,
    period_index,
    validate_columns,
    validate_period_partition,
    validate_periods,
)

# closure_of is renamed out of the way before the join, so a column name
# present in both frames (start_time, the grouping columns, ...) stays
# unambiguous afterwards. Only closure_from columns survive the semi join.
_OF_PREFIX = "_ts_of_"


def _proximity_condition(
        start_time_column: str,
        end_time_column: str,
        max_interval_seconds: int,
) -> Column:
    """True when the two segments overlap or lie within max_interval_seconds.

    The gap between closure_from's [s1, e1] and closure_of's [s2, e2] is at
    most `max` exactly when s1 <= e2 + max and e1 >= s2 - max; overlapping
    segments satisfy both.
    """
    from_start = F.col(f"`{start_time_column}`").cast("long")
    from_end = F.col(f"`{end_time_column}`").cast("long")
    of_start = F.col(f"`{_OF_PREFIX}{start_time_column}`").cast("long")
    of_end = F.col(f"`{_OF_PREFIX}{end_time_column}`").cast("long")
    return ((from_start <= of_end + max_interval_seconds)
            & (from_end >= of_start - max_interval_seconds))


def _neighbouring_periods(index: Column) -> Column:
    """A period index and its two neighbours, as an array.

    Nothing further out can ever be proximate: a segment ends at the latest on
    its own period border, so a row two periods away is at least one whole
    period distant, and max_interval_seconds is strictly shorter than that.
    """
    return F.array(index - 1, index, index + 1)


def _relevant_from_rows(
        closure_of: DataFrame,
        closure_from: DataFrame,
        start_time_column: str,
        end_time_column: str,
        period: PeriodPartition,
) -> DataFrame:
    """The closure_from rows that could take part in the closure at all.

    Only a row in a period held by closure_of, or in one on either side of it,
    can ever be proximate (see compute_closure_by_time_proximity); the rest can
    never be selected, so they are not held to the single-period rule.
    """
    of_periods = closure_of.select(
        period_index(start_time_column, period).alias(PERIOD_INDEX)).distinct()
    neighbouring = of_periods.select(
        F.explode(_neighbouring_periods(F.col(PERIOD_INDEX))).alias(PERIOD_INDEX)
    ).distinct()

    # Either bound landing in a relevant period makes the row relevant, so a
    # row crossing into one is still checked.
    touches = ((period_index(start_time_column, period) == F.col(PERIOD_INDEX))
               | (period_index(end_time_column, period) == F.col(PERIOD_INDEX)))
    return closure_from.join(F.broadcast(neighbouring), touches, "left_semi")


def compute_closure_by_time_proximity(
        closure_of: DataFrame,
        closure_from: DataFrame,
        group_by_columns: list[str],
        max_interval_seconds: int,
        start_time_column: str,
        end_time_column: str,
        period: PeriodPartition = None,
) -> DataFrame:
    """The closure_from rows that are close in time to a closure_of row.

    A closure_from row is in the closure when some closure_of row agrees with
    it on every column in `group_by_columns` and their segments overlap or lie
    no more than max_interval_seconds apart. Only those three things decide the
    result: the grouping columns and the two time bounds.

    The rows come back exactly as they are in closure_from, with their own
    columns and nothing added. Being close to several closure_of rows does not
    duplicate a row -- each one is returned once, or not at all.

    Args:
    - closure_of: the rows whose neighbourhood is wanted. Only its
      group_by_columns and time bounds are read; it contributes no columns to
      the result.
    - closure_from: the rows to select from, and the shape of the result.
    - group_by_columns: columns that must agree; present in both frames. Null
      matches null, as it does elsewhere in this codebase.
    - max_interval_seconds: how far apart two segments may be and still count
      as close; 0 means they must overlap or touch.
    - start_time_column, end_time_column: the segment bounds, named the same
      in both frames.
    - period: purely an optimization -- it does not change the result. It
      turns the proximity test from a range join into an equality join: since
      max_interval_seconds is shorter than a period, a proximate row is always
      in the same period or in the one before or after, so each closure_of row
      is offered to those three periods and matched by period index. This
      requires max_interval_seconds < period.min_seconds, and each segment to
      lie inside a single period, which is validated: closure_of in full, and
      closure_from only where it could matter -- the periods held by
      closure_of and their immediate neighbours.

    Returns:
    - DataFrame: the closure_from rows in the closure, with closure_from's
      columns.

    Raises:
    - ValueError: a referenced column is missing from either frame, the period
      is not longer than max_interval_seconds, or a segment of closure_of (or
      a relevant segment of closure_from) is not inside a single period.
    """
    if not group_by_columns:
        raise ValueError("group_by_columns must not be empty")
    if max_interval_seconds < 0:
        raise ValueError(f"max_interval_seconds must be >= 0, got {max_interval_seconds}")

    referenced = [*group_by_columns, start_time_column, end_time_column]
    validate_columns(closure_of, referenced, "closure_of")
    validate_columns(closure_from, referenced, "closure_from")

    if period is not None:
        validate_period_partition(period, max_interval_seconds,
                                  "to use the period for proximity")
        validate_periods(closure_of, start_time_column, end_time_column, period,
                         subject="closure_of segment")
        validate_periods(
            _relevant_from_rows(closure_of, closure_from, start_time_column,
                                end_time_column, period),
            start_time_column, end_time_column, period, subject="closure_from segment",
        )

    of = closure_of.select(
        [F.col(f"`{column}`").alias(f"{_OF_PREFIX}{column}") for column in closure_of.columns])

    condition = (
        null_safe_condition(group_by_columns,
                            [f"{_OF_PREFIX}{column}" for column in group_by_columns])
        & _proximity_condition(start_time_column, end_time_column, max_interval_seconds)
    )
    if period is not None:
        # The two rows must also sit in neighbouring periods. Written directly
        # that is abs(of period - from period) <= 1, a range comparison; giving
        # each closure_of row a copy per period it reaches turns it into the
        # equality below, which the planner can hash-join on. Offering the
        # closure_of side is also what keeps the result unduplicated: the semi
        # join returns a closure_from row once however many copies it meets.
        of = of.withColumn(
            PERIOD_INDEX,
            F.explode(_neighbouring_periods(
                period_index(f"{_OF_PREFIX}{start_time_column}", period))),
        )
        in_a_neighbouring_period = (
            F.col(PERIOD_INDEX) == period_index(start_time_column, period))
        condition = condition & in_a_neighbouring_period

    return closure_from.join(of, condition, "left_semi")
