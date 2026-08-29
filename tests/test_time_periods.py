"""Unit tests for time_periods.py, the code shared by time_stretch and
time_closure. See conftest.py for the fixtures and frame helpers."""

from datetime import datetime

import pytest
import pyspark.sql.functions as F
from pyspark.sql.window import Window

from conftest import SIX_HOURS, at, frame, hhmm, rows_of
from time_periods import (
    PeriodPartition,
    agg_name,
    agg_output_names,
    build_agg_expressions,
    null_safe_join,
    period_index,
    period_start,
    split_on_period_borders,
    validate_aggs,
    validate_columns,
    validate_period_column,
    validate_period_partition,
    validate_periods,
)


# --- PeriodPartition ---------------------------------------------------------

def test_members_are_only_the_four_periods():
    """COLUMN is a shared constant, not a fifth member."""
    assert [m.name for m in PeriodPartition] == ["HOUR", "DAY", "MONTH", "YEAR"]
    assert PeriodPartition.COLUMN == "et"
    assert all(m.column == "et" for m in PeriodPartition)


@pytest.mark.parametrize(
    "period, fmt, unit, min_seconds",
    [
        (PeriodPartition.HOUR, "yyyyMMddHH", "hour", 3_600),
        (PeriodPartition.DAY, "yyyyMMdd", "day", 86_400),
        (PeriodPartition.MONTH, "yyyyMM", "month", 28 * 86_400),
        (PeriodPartition.YEAR, "yyyy", "year", 365 * 86_400),
    ],
)
def test_member_fields(period, fmt, unit, min_seconds):
    """size is 1 for the plain calendar periods, so min_seconds is one unit."""
    assert (period.fmt, period.unit, period.size) == (fmt, unit, 1)
    assert period.min_seconds == min_seconds


def test_size_multiplies_min_seconds():
    """A partition of `size` units is `size` times as long."""
    assert SIX_HOURS.min_seconds == 6 * PeriodPartition.HOUR.min_seconds


@pytest.mark.parametrize(
    "period, minutes",
    [
        (PeriodPartition.HOUR, 60),
        (PeriodPartition.DAY, 24 * 60),
        (SIX_HOURS, 6 * 60),
    ],
)
def test_interval_advances_by_one_partition(spark_session, period, minutes):
    df = frame(spark_session, ["t"], [(at(0),)])
    advanced = df.select((F.col("t") + period.interval).alias("t"))
    assert rows_of(advanced) == [(hhmm(minutes),)]


# --- agg_name / agg_output_names ---------------------------------------------

@pytest.mark.parametrize(
    "func, expected",
    [(F.sum, "sum"), (F.min, "min"), (F.max, "max"), (F.avg, "avg"), (F.count, "count")],
)
def test_agg_name(func, expected):
    assert agg_name(func) == expected


def test_agg_output_names():
    assert agg_output_names([(F.sum, ["bytes"]), (F.max, ["bytes", "score"])]) == [
        "bytes_sum", "bytes_max", "score_max",
    ]


# --- period_index / period_start ---------------------------------------------

@pytest.mark.parametrize(
    "period, unit",
    [
        (PeriodPartition.HOUR, "hour"),
        (PeriodPartition.DAY, "day"),
        (PeriodPartition.MONTH, "month"),
        (PeriodPartition.YEAR, "year"),
    ],
)
def test_period_start_matches_date_trunc(spark_session, period, unit):
    """For size 1 the period start is exactly Spark's own date_trunc, checked
    over 15 months of timestamps at an odd stride (so every hour-of-day, every
    day-of-week and both sides of a month/year border are hit)."""
    probe = spark_session.sql(
        "SELECT explode(sequence(TIMESTAMP '2025-11-30 22:15:00', "
        "TIMESTAMP '2027-02-02 03:00:00', INTERVAL 7 HOUR)) AS t"
    )
    disagreeing = (
        probe.withColumn("mine", period_start("t", period))
        .withColumn("theirs", F.date_trunc(unit, F.col("t")))
        .filter(F.col("mine") != F.col("theirs"))
    )
    assert disagreeing.count() == 0


@pytest.mark.parametrize(
    "period, same, different",
    [
        # (period, two timestamps in one period, one in the next)
        (PeriodPartition.HOUR, (at(0), at(59)), at(60)),
        (PeriodPartition.DAY, (at(0), at(24 * 60 - 1)), at(24 * 60)),
        (PeriodPartition.MONTH, (datetime(2026, 3, 1), datetime(2026, 3, 31, 23, 59)),
         datetime(2026, 4, 1)),
        (PeriodPartition.YEAR, (datetime(2026, 1, 1), datetime(2026, 12, 31, 23, 59)),
         datetime(2027, 1, 1)),
        (SIX_HOURS, (at(0), at(6 * 60 - 1)), at(6 * 60)),
    ],
)
def test_period_index_identifies_the_period(spark_session, period, same, different):
    df = frame(spark_session, ["a", "b", "c"], [(same[0], same[1], different)])
    indices = df.select(
        period_index("a", period).alias("a"),
        period_index("b", period).alias("b"),
        period_index("c", period).alias("c"),
    ).collect()[0]
    assert indices["a"] == indices["b"]
    assert indices["c"] == indices["a"] + 1


def test_period_index_is_consecutive_across_a_year_border(spark_session):
    df = frame(spark_session, ["dec", "jan"],
               [(datetime(2026, 12, 31, 23, 0), datetime(2027, 1, 1, 0, 30))])
    for period in (PeriodPartition.HOUR, PeriodPartition.DAY, PeriodPartition.MONTH):
        row = df.select(period_index("dec", period).alias("d"),
                        period_index("jan", period).alias("j")).collect()[0]
        assert row["j"] == row["d"] + 1, period.name


def test_period_start_buckets_by_size(spark_session):
    """With size 6 the day splits into partitions starting at 00, 06, 12, 18."""
    times = [(at(minutes),) for minutes in (0, 5 * 60, 6 * 60, 11 * 60, 12 * 60, 23 * 60)]
    df = frame(spark_session, ["t"], times)
    starts = df.select(period_start("t", SIX_HOURS).alias("t"))
    assert rows_of(starts) == [
        (hhmm(0),), (hhmm(0),), (hhmm(6 * 60),),
        (hhmm(6 * 60),), (hhmm(12 * 60),), (hhmm(18 * 60),),
    ]


# --- split_on_period_borders -------------------------------------------------

def _split(spark, period, rows, columns=("name", "start_time", "end_time")):
    df = frame(spark, list(columns), rows)
    return split_on_period_borders(df, "start_time", "end_time", period)


def test_split_leaves_a_contained_segment_alone(spark_session):
    split = _split(spark_session, PeriodPartition.HOUR, [("s", at(10), at(50))])
    assert rows_of(split) == [("s", hhmm(10), hhmm(50), "2026010100")]


def test_split_cuts_a_segment_spanning_three_periods(spark_session):
    split = _split(spark_session, PeriodPartition.HOUR, [("s", at(90), at(200))])
    assert rows_of(split, ["start_time"]) == [
        ("s", hhmm(90), hhmm(120), "2026010101"),
        ("s", hhmm(120), hhmm(180), "2026010102"),
        ("s", hhmm(180), hhmm(200), "2026010103"),
    ]


def test_split_does_not_emit_an_empty_tail(spark_session):
    """A segment ending exactly on a border belongs to the period it filled."""
    split = _split(spark_session, PeriodPartition.HOUR, [("s", at(30), at(60))])
    assert rows_of(split) == [("s", hhmm(30), hhmm(60), "2026010100")]


def test_split_keeps_a_zero_length_segment(spark_session):
    split = _split(spark_session, PeriodPartition.HOUR, [("s", at(30), at(30))])
    assert rows_of(split) == [("s", hhmm(30), hhmm(30), "2026010100")]


def test_split_keeps_a_zero_length_segment_on_a_border(spark_session):
    split = _split(spark_session, PeriodPartition.HOUR, [("s", at(60), at(60))])
    assert rows_of(split) == [("s", hhmm(60), hhmm(60), "2026010101")]


def test_split_on_day_borders_across_a_month(spark_session):
    split = _split(spark_session, PeriodPartition.DAY,
                   [("s", datetime(2026, 1, 30, 22), datetime(2026, 2, 2, 1))])
    assert rows_of(split, ["start_time"]) == [
        ("s", "30 22:00", "31 00:00", "20260130"),
        ("s", "31 00:00", "01 00:00", "20260131"),
        ("s", "01 00:00", "02 00:00", "20260201"),
        ("s", "02 00:00", "02 01:00", "20260202"),
    ]


def test_split_on_month_borders_across_a_year(spark_session):
    split = _split(spark_session, PeriodPartition.MONTH,
                   [("s", datetime(2026, 11, 15), datetime(2027, 1, 20))])
    assert [row[3] for row in rows_of(split, ["start_time"])] == ["202611", "202612", "202701"]


def test_split_on_year_borders(spark_session):
    split = _split(spark_session, PeriodPartition.YEAR,
                   [("s", datetime(2025, 6, 1), datetime(2027, 6, 1))])
    assert [row[3] for row in rows_of(split, ["start_time"])] == ["2025", "2026", "2027"]


def test_split_uses_the_period_size(spark_session):
    """With size 6 a segment is only cut at 00/06/12/18, not every hour."""
    split = _split(spark_session, SIX_HOURS, [("s", at(5 * 60), at(13 * 60))])
    assert rows_of(split, ["start_time"]) == [
        ("s", hhmm(5 * 60), hhmm(6 * 60), "2026010100"),
        ("s", hhmm(6 * 60), hhmm(12 * 60), "2026010106"),
        ("s", hhmm(12 * 60), hhmm(13 * 60), "2026010112"),
    ]


def test_split_repeats_the_other_columns_on_every_piece(spark_session):
    split = _split(spark_session, PeriodPartition.HOUR, [("s", at(30), at(150), 42)],
                   columns=("name", "start_time", "end_time", "bytes_sum"))
    assert [(row[0], row[3]) for row in rows_of(split, ["start_time"])] == [
        ("s", 42), ("s", 42), ("s", 42),
    ]


def test_split_overwrites_an_existing_period_column(spark_session):
    """A piece is labelled with its own period, not the one it came from."""
    df = frame(spark_session, ["start_time", "end_time", "et"],
               [(at(30), at(90), "2026010100")])
    split = split_on_period_borders(df, "start_time", "end_time", PeriodPartition.HOUR)
    assert split.columns == ["start_time", "end_time", "et"]
    assert rows_of(split, ["start_time"]) == [
        (hhmm(30), hhmm(60), "2026010100"),
        (hhmm(60), hhmm(90), "2026010101"),
    ]


def test_split_pieces_cover_the_original(spark_session):
    rows = [("s%d" % i, at(start), at(start + length))
            for i, (start, length) in enumerate([(0, 5), (30, 200), (119, 2), (240, 1000)])]
    split = _split(spark_session, PeriodPartition.HOUR, rows)
    covered = (split.groupBy("name")
               .agg(F.min("start_time").alias("start_time"),
                    F.max("end_time").alias("end_time")))
    assert rows_of(covered, ["name"]) == [
        (name, start.strftime("%d %H:%M"), end.strftime("%d %H:%M"))
        for name, start, end in rows
    ]


def test_split_pieces_are_contiguous(spark_session):
    """Consecutive pieces meet exactly: no gap between them, no overlap. The
    covering test above would still pass with either."""
    rows = [("s%d" % i, at(start), at(start + length))
            for i, (start, length) in enumerate([(0, 5), (30, 200), (119, 2), (240, 1000)])]
    split = _split(spark_session, PeriodPartition.HOUR, rows)
    ordered = Window.partitionBy("name").orderBy("start_time")
    breaks = (split
              .withColumn("next_start", F.lead("start_time").over(ordered))
              .filter(F.col("next_start").isNotNull()
                      & (F.col("next_start") != F.col("end_time"))))
    assert breaks.count() == 0, rows_of(breaks, ["name", "start_time"])


# --- null_safe_join ----------------------------------------------------------

def test_null_safe_join_matches_null_keys(spark_session):
    left = frame(spark_session, ["key", "value"], [("a", 1), (None, 2)])
    right = frame(spark_session, ["key", "extra"], [("a", "x"), (None, "y")])
    joined = null_safe_join(left, right, ["key"]).select("l.value", "r.extra")
    assert sorted(rows_of(joined)) == [(1, "x"), (2, "y")]


# --- validate_columns --------------------------------------------------------

def test_validate_columns_accepts_present_columns(spark_session):
    df = frame(spark_session, ["a", "b"], [(1, 2)])
    validate_columns(df, ["a", "b"])


def test_validate_columns_reports_every_missing_one(spark_session):
    df = frame(spark_session, ["a", "b"], [(1, 2)])
    with pytest.raises(ValueError) as error:
        validate_columns(df, ["a", "x", "y"], "closure_from")
    message = str(error.value)
    assert "columns not found in closure_from" in message
    assert "'x'" in message and "'y'" in message and "'a'" not in message.split(";")[0]


# --- validate_aggs / build_agg_expressions -----------------------------------

def test_validate_aggs_returns_the_referenced_columns():
    assert validate_aggs([(F.sum, ["a"]), (F.max, ["a", "b"])]) == ["a", "a", "b"]


@pytest.mark.parametrize(
    "aggs, message",
    [
        pytest.param([(F.sum,)], "must be a (function, [columns]) pair", id="not_a_pair"),
        pytest.param([("sum", ["a"])], "is not callable", id="function_not_callable"),
        pytest.param([(F.sum, "a")], "must be a non-empty list", id="columns_as_string"),
        pytest.param([(F.sum, [])], "must be a non-empty list", id="columns_empty"),
        pytest.param([(F.sum, [1])], "must all be column names", id="column_not_a_name"),
    ],
)
def test_validate_aggs_rejects(aggs, message):
    with pytest.raises(ValueError) as error:
        validate_aggs(aggs)
    assert message in str(error.value)


def test_build_agg_expressions_names_and_order(spark_session):
    df = frame(spark_session, ["key", "bytes"], [("a", 1)])
    exprs = build_agg_expressions([(F.sum, ["bytes"]), (F.max, ["bytes", "key"])], [])
    assert df.groupBy().agg(*exprs).columns == ["bytes_sum", "bytes_max", "key_max"]


def test_build_agg_expressions_reads_the_prefixed_column(spark_session):
    """The prefix moves the read to the renamed side of a join; the output
    column keeps the plain name."""
    df = frame(spark_session, ["from_bytes"], [(3,), (4,)])
    exprs = build_agg_expressions([(F.sum, ["bytes"])], [], source_prefix="from_")
    assert rows_of(df.groupBy().agg(*exprs)) == [(7,)]
    assert df.groupBy().agg(*exprs).columns == ["bytes_sum"]


def test_build_agg_expressions_rejects_a_duplicate_output():
    with pytest.raises(ValueError, match="produced more than once"):
        build_agg_expressions([(F.sum, ["bytes"]), (F.sum, ["bytes"])], [])


def test_build_agg_expressions_rejects_a_reserved_name():
    with pytest.raises(ValueError, match="produced more than once"):
        build_agg_expressions([(F.sum, ["bytes"])], ["bytes_sum"])


def test_build_agg_expressions_rejects_a_function_that_raises():
    with pytest.raises(ValueError, match="failed on column"):
        build_agg_expressions([(len, ["bytes"])], [])


def test_build_agg_expressions_rejects_a_function_returning_non_column():
    with pytest.raises(ValueError, match="must be a Spark aggregation"):
        build_agg_expressions([(lambda column: "not a column", ["bytes"])], [])


# --- validate_period_partition / validate_period_column ----------------------

def test_validate_period_partition_rejects_a_non_member():
    with pytest.raises(ValueError, match="must be a PeriodPartition"):
        validate_period_partition("HOUR", 1, "for testing")


def test_validate_period_partition_bound_is_strict():
    """A gap of exactly one period is too long, for every caller."""
    validate_period_partition(PeriodPartition.HOUR, 3_599, "for testing")
    with pytest.raises(ValueError, match="must be shorter than one HOUR"):
        validate_period_partition(PeriodPartition.HOUR, 3_600, "for testing")
    with pytest.raises(ValueError, match="must be shorter than one DAY"):
        validate_period_partition(PeriodPartition.DAY, 86_400, "for testing")


def test_validate_period_column(spark_session):
    with_column = frame(spark_session, ["et", "x"], [("2026010100", 1)])
    validate_period_column(with_column, PeriodPartition.HOUR)
    without = frame(spark_session, ["x"], [(1,)])
    with pytest.raises(ValueError, match="expects the partition column"):
        validate_period_column(without, PeriodPartition.HOUR)


# --- validate_periods --------------------------------------------------------

def _bounds(spark, pairs):
    return frame(spark, ["start_time", "end_time"], pairs)


def test_validate_periods_accepts_contained_segments(spark_session):
    df = _bounds(spark_session, [(at(0), at(59)), (at(61), at(119)), (at(0), at(0))])
    validate_periods(df, "start_time", "end_time", PeriodPartition.HOUR)


def test_validate_periods_rejects_a_segment_crossing_a_border(spark_session):
    df = _bounds(spark_session, [(at(0), at(10)), (at(55), at(65))])
    with pytest.raises(ValueError, match="spans more than one HOUR period"):
        validate_periods(df, "start_time", "end_time", PeriodPartition.HOUR)
    # the same segment is inside a single DAY
    validate_periods(df, "start_time", "end_time", PeriodPartition.DAY)


def test_validate_periods_rejects_a_reversed_segment(spark_session):
    df = _bounds(spark_session, [(at(30), at(10))])
    with pytest.raises(ValueError, match="ends before it starts"):
        validate_periods(df, "start_time", "end_time", PeriodPartition.HOUR)


def test_validate_periods_rejects_a_null_bound(spark_session):
    df = _bounds(spark_session, [(at(0), at(10))]).withColumn(
        "end_time", F.lit(None).cast("timestamp"))
    with pytest.raises(ValueError, match="has a null bound"):
        validate_periods(df, "start_time", "end_time", PeriodPartition.HOUR)


def test_validate_periods_names_the_offending_segment(spark_session):
    """Reported in the session timezone, not the driver's."""
    df = _bounds(spark_session, [(at(0), at(10)), (at(55), at(65))])
    with pytest.raises(ValueError) as error:
        validate_periods(df, "start_time", "end_time", PeriodPartition.HOUR)
    assert "00:55:00" in str(error.value) and "01:05:00" in str(error.value)


def test_validate_periods_uses_the_subject_in_the_message(spark_session):
    df = _bounds(spark_session, [(at(55), at(65))])
    with pytest.raises(ValueError, match="closure_from segment"):
        validate_periods(df, "start_time", "end_time", PeriodPartition.HOUR,
                         subject="closure_from segment")
