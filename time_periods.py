"""Period partitioning and validation shared by time_stretch and time_closure.

Both modules work on the same shape of data -- rows carrying a time segment
[start, end] plus grouping columns -- and both can use a PeriodPartition to
turn expensive whole-timeline work into partition-sized work. Everything they
have in common lives here.
"""

from enum import Enum
from functools import reduce
from operator import and_
from typing import Callable, Sequence

import pyspark.sql.functions as F
from pyspark.sql import Column, DataFrame


class PeriodPartition(Enum):
    """A calendar period used to partition a timestamp column.

    A member's value is (fmt, size, unit, one_period_min_seconds):
    - fmt: the Spark ``date_format`` pattern that maps the period's start to
      its partition value (e.g. 2026-08-28 14:05 -> "2026082814" for HOUR).
    - size: how many `unit`s make up one partition. size=1 is the plain
      calendar period; SIX_HOURS = ("yyyyMMddHH", 6, "hour", 3_600) would
      bucket the day into four six-hour partitions.
    - unit: the calendar unit the period is counted in.
    - one_period_min_seconds: the shortest a single `unit` can be. Months and
      years vary in length, so this is February / a non-leap year.

    and exposes:
    - min_seconds: the shortest the whole partition can be (unit * size).
    - column: the DataFrame column holding the partition value.

    Borders are calendar borders in UTC (no DST), matching
    ``spark.sql.session.timeZone = UTC``.
    """

    HOUR = ("yyyyMMddHH", 1, "hour", 3_600)
    DAY = ("yyyyMMdd", 1, "day", 86_400)
    MONTH = ("yyyyMM", 1, "month", 28 * 86_400)
    YEAR = ("yyyy", 1, "year", 365 * 86_400)

    def __init__(self, fmt: str, size: int, unit: str, one_period_min_seconds: int):
        self.fmt = fmt
        self.size = size
        self.unit = unit
        self.min_seconds = one_period_min_seconds * size

    @property
    def column(self) -> str:
        return PeriodPartition.COLUMN

    @property
    def interval(self) -> Column:
        """One partition as a Spark interval, e.g. INTERVAL 6 HOUR."""
        return F.expr(f"INTERVAL {self.size} {self.unit.upper()}")


# The DataFrame column holding the partition value, shared by every member.
# Assigned after the class body: inside it, any plain assignment would become
# a fifth enum member (and enum.nonmember needs Python 3.11+).
PeriodPartition.COLUMN = "et"


# An aggregation is a Spark aggregate function -- F.max, F.min, F.sum, ... --
# paired with the columns it is applied to.
AggFunction = Callable[[Column], Column]
Aggregation = tuple[AggFunction, list[str]]

# Internal helper columns. Prefixed so they cannot collide with user columns.
PERIOD_INDEX = "_ts_period_index"
_PERIOD_START = "_ts_period_start"
_PIECE_START = "_ts_piece_start"
_PIECE_END = "_ts_piece_end"
_START_TEXT = "_ts_start_text"
_END_TEXT = "_ts_end_text"

_SECONDS_PER_UNIT = {"hour": 3_600, "day": 86_400}


# --- periods -----------------------------------------------------------------

def period_index(time_column: str, period: PeriodPartition) -> Column:
    """The index of the partition holding `time_column`: consecutive periods
    get consecutive indices, so two timestamps share a period iff their
    indices are equal, and neighbouring periods differ by one."""
    column = F.col(time_column)
    if period.unit in _SECONDS_PER_UNIT:
        # Epoch-based, so the borders are UTC midnight / o-clock.
        return F.floor(column.cast("long") / (_SECONDS_PER_UNIT[period.unit] * period.size))
    if period.unit == "month":
        months = F.year(column) * 12 + F.month(column) - 1
        return F.floor(months / period.size)
    if period.unit == "year":
        return F.floor(F.year(column) / period.size)
    raise ValueError(f"unsupported period unit {period.unit!r}")


def period_start(time_column: str, period: PeriodPartition) -> Column:
    """The first instant of the partition holding `time_column`."""
    index = period_index(time_column, period)
    if period.unit in _SECONDS_PER_UNIT:
        return F.timestamp_seconds(index * _SECONDS_PER_UNIT[period.unit] * period.size)
    if period.unit == "month":
        months = index * period.size
        return F.make_date(
            F.floor(months / 12).cast("int"),
            (F.pmod(months, F.lit(12)) + 1).cast("int"),
            F.lit(1),
        ).cast("timestamp")
    if period.unit == "year":
        return F.make_date((index * period.size).cast("int"), F.lit(1), F.lit(1)).cast("timestamp")
    raise ValueError(f"unsupported period unit {period.unit!r}")


def split_on_period_borders(
        df: DataFrame,
        start_time_column: str,
        end_time_column: str,
        period: PeriodPartition,
) -> DataFrame:
    """Cut every segment that crosses a period border into one row per period.

    A segment covering three periods becomes three rows, each clipped to its
    own period: [start, border1], [border1, border2], [border2, end]. Segments
    already inside one period pass through unchanged. Every other column is
    carried onto each piece as it is -- an aggregated value is repeated on the
    pieces, not re-divided between them.

    The period column (`period.column`) is (re)computed from each piece, so
    the result is partitioned consistently.

    Args:
    - df: the segments.
    - start_time_column, end_time_column: the segment bounds, timestamps.
    - period: the partitioning whose borders to cut on.

    Returns:
    - DataFrame: the input columns, plus period.column, one row per
      (segment, period) pair.
    """
    step = period.interval
    # Every period the segment touches, as a row apiece.
    covered = F.sequence(
        period_start(start_time_column, period),
        period_start(end_time_column, period),
        step,
    )
    piece_start = F.greatest(F.col(start_time_column), F.col(_PERIOD_START))
    piece_end = F.least(F.col(end_time_column), F.col(_PERIOD_START) + step)

    return (
        df.withColumn(_PERIOD_START, F.explode(covered))
        .withColumn(_PIECE_START, piece_start)
        .withColumn(_PIECE_END, piece_end)
        # A segment ending exactly on a border reaches into the next period
        # for zero seconds; drop that empty tail, but keep a genuine instant.
        .filter(
            (F.col(_PIECE_START) < F.col(_PIECE_END))
            | (F.col(start_time_column) == F.col(end_time_column))
        )
        .withColumn(period.column, F.date_format(F.col(_PERIOD_START), period.fmt))
        .withColumn(start_time_column, F.col(_PIECE_START))
        .withColumn(end_time_column, F.col(_PIECE_END))
        .drop(_PERIOD_START, _PIECE_START, _PIECE_END)
    )


# --- joining -----------------------------------------------------------------

def null_safe_condition(left_columns: Sequence[str], right_columns: Sequence[str]) -> Column:
    """`left.a <=> right.a AND ...`, matching null against null (a plain
    equality would silently drop rows whose grouping value is null)."""
    return reduce(
        and_,
        (F.col(f"`{left}`").eqNullSafe(F.col(f"`{right}`"))
         for left, right in zip(left_columns, right_columns)),
    )


def null_safe_join(left: DataFrame, right: DataFrame, keys: Sequence[str]) -> DataFrame:
    """Inner join on `keys`, matching null against null."""
    left, right = left.alias("l"), right.alias("r")
    condition = reduce(
        and_, (F.col(f"l.`{key}`").eqNullSafe(F.col(f"r.`{key}`")) for key in keys)
    )
    return left.join(right, condition, "inner")


# --- validation --------------------------------------------------------------

def agg_name(func: Callable) -> str:
    """Output suffix for an aggregation function: F.sum -> "sum"."""
    return getattr(func, "__name__", str(func)).rstrip("_")


def validate_columns(df: DataFrame, columns: Sequence[str], what: str = "the DataFrame") -> None:
    """Fail if any of `columns` is missing, naming all the missing ones."""
    available = set(df.columns)
    missing = [c for c in dict.fromkeys(columns) if c not in available]
    if missing:
        raise ValueError(
            f"columns not found in {what}: {missing}; available: {sorted(available)}"
        )


def validate_aggs(aggs: Sequence[Aggregation]) -> list[str]:
    """Check the shape of every aggs entry and return the columns they read."""
    columns = []
    for entry in aggs:
        if not (isinstance(entry, (tuple, list)) and len(entry) == 2):
            raise ValueError(f"each aggs entry must be a (function, [columns]) pair, got {entry!r}")
        func, entry_columns = entry
        if not callable(func):
            raise ValueError(f"aggs function {func!r} is not callable")
        if isinstance(entry_columns, str) or not entry_columns:
            raise ValueError(
                f"aggs columns for {agg_name(func)} must be a non-empty list, got {entry_columns!r}")
        if not all(isinstance(column, str) for column in entry_columns):
            raise ValueError(
                f"aggs columns for {agg_name(func)} must all be column names, got {entry_columns!r}")
        columns.extend(entry_columns)
    return columns


def agg_output_names(aggs: Sequence[Aggregation]) -> list[str]:
    """The column each aggregation produces: (F.sum, ["bytes"]) -> bytes_sum."""
    return [f"{column}_{agg_name(func)}" for func, columns in aggs for column in columns]


def build_agg_expressions(
        aggs: Sequence[Aggregation],
        reserved_names: Sequence[str],
        source_prefix: str = "",
) -> list[Column]:
    """The aggregation expressions, named "{column}_{function}".

    `source_prefix` is prepended when reading the input column, for callers
    that renamed one side of a join out of the way; the output name is
    unprefixed either way. `reserved_names` are names already taken by the
    output, which an aggregation may not shadow.
    """
    produced = set(reserved_names)
    expressions = []
    names = agg_output_names(aggs)
    pairs = [(func, column) for func, columns in aggs for column in columns]
    for name, (func, column) in zip(names, pairs):
        if name in produced:
            raise ValueError(f"aggregation output column {name!r} is produced more than once")
        produced.add(name)
        # F.max(col) & co. build a Column; anything that does not (a plain
        # lambda, a non-aggregate helper) is rejected here rather than deep
        # inside the groupBy.
        try:
            expression = func(F.col(f"`{source_prefix}{column}`"))
        except Exception as exc:
            raise ValueError(f"aggs function {agg_name(func)} failed on column {column!r}: {exc}") from exc
        if not isinstance(expression, Column):
            raise ValueError(
                f"aggs function {agg_name(func)} must be a Spark aggregation such as "
                f"F.max / F.min / F.sum; applied to {column!r} it returned {expression!r}"
            )
        expressions.append(expression.alias(name))
    return expressions


def validate_period_partition(
        period,
        max_interval_seconds: int,
        purpose: str,
) -> None:
    """Check the period argument itself against max_interval_seconds.

    The gap has to be strictly shorter than a period. That is what lets a
    caller reason about a bounded number of periods: at exactly one period,
    two segments in periods two apart -- the first ending on its border, the
    second starting on its own -- would still be within the gap.

    `purpose` completes the error message with what the period was for.
    """
    if not isinstance(period, PeriodPartition):
        raise ValueError(f"period must be a PeriodPartition, got {period!r}")
    if max_interval_seconds >= period.min_seconds:
        raise ValueError(
            f"max_interval_seconds ({max_interval_seconds}) must be shorter than one "
            f"{period.name} ({period.min_seconds}s) {purpose}"
        )


def validate_period_column(df: DataFrame, period: PeriodPartition, what: str = "the DataFrame") -> None:
    """Fail unless the partition column the period reads is present."""
    if period.column not in set(df.columns):
        raise ValueError(
            f"period {period.name} expects the partition column {period.column!r}, "
            f"which is not in {what}; available: {sorted(df.columns)}"
        )


def validate_periods(
        df: DataFrame,
        start_time_column: str,
        end_time_column: str,
        period: PeriodPartition,
        subject: str = "segment",
) -> None:
    """Fail unless every segment is a sane interval inside a single period.

    Runs one short-circuiting Spark job: it stops at the first offending row.
    """
    start, end = F.col(start_time_column), F.col(end_time_column)
    # A segment ending exactly on a border fills the earlier period and
    # occupies none of the next one, which is how split_on_period_borders cuts
    # it too -- so it counts as the earlier period, and a stretched segment can
    # be fed straight back in. A zero-length segment on a border is exempt:
    # there is no earlier period for it to belong to.
    ends_on_border = (end == period_start(end_time_column, period)) & (end > start)
    end_index = F.when(ends_on_border,
                       period_index(end_time_column, period) - 1
                       ).otherwise(period_index(end_time_column, period))
    # The bounds are also carried as text, formatted by Spark: collect() would
    # otherwise report them in the driver timezone rather than the session one.
    readable = "yyyy-MM-dd HH:mm:ss"
    offending = (
        df.select(
            start_time_column, end_time_column,
            F.date_format(start, readable).alias(_START_TEXT),
            F.date_format(end, readable).alias(_END_TEXT),
        )
        .filter(
            start.isNull() | end.isNull() | (end < start)
            | (period_index(start_time_column, period) != end_index)
        )
        .limit(1)
        .collect()
    )
    if not offending:
        return
    row = offending[0]
    values = f"[{row[_START_TEXT] or 'null'}, {row[_END_TEXT] or 'null'}]"
    if row[start_time_column] is None or row[end_time_column] is None:
        problem = "has a null bound"
    elif row[end_time_column] < row[start_time_column]:
        problem = "ends before it starts"
    else:
        problem = f"spans more than one {period.name} period"
    raise ValueError(
        f"{subject} {values} {problem}; with period={period.name} every segment must lie "
        f"inside a single period -- run split_on_period_borders() on the input first"
    )
