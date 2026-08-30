"""Unit tests for time_closure.py. See conftest.py for the fixtures and frame
helpers, and test_time_periods.py for the shared validation code."""

import random

import pytest
import pyspark.sql.functions as F

from tests.conftest import at, frame, hhmm, rows_of, segments
from time_closure import compute_closure_by_time_proximity
from time_periods import PeriodPartition

CLOSURE_COLUMNS = ["key", "start_time", "end_time", "name"]


def _rows(spark, rows, with_period=None):
    """Rows of (key, start_minutes, end_minutes, name), optionally with the
    partition column filled in from the start time."""
    columns = list(CLOSURE_COLUMNS)
    # SQL VALUES needs at least one row, so an empty frame is a typed row
    # filtered away.
    built = [(key, at(start), at(end), name)
             for key, start, end, name in (rows or [("x", 0, 0, "x")])]
    if with_period is not None:
        columns.append(with_period.column)
        built = [row + (row[1].strftime("%Y%m%d%H"),) for row in built]
    df = frame(spark, columns, built)
    return df if rows else df.filter(F.lit(False))


def _closure(spark, of, frm, max_interval_seconds=300, period=None,
             group_by_columns=("key",)):
    return compute_closure_by_time_proximity(
        _rows(spark, of, period), _rows(spark, frm, period), list(group_by_columns),
        max_interval_seconds, "start_time", "end_time", period,
    )


def _names(result):
    """The `name` of every selected row, sorted."""
    return sorted(row["name"] for row in result.collect())


# --- what counts as close ----------------------------------------------------

# (test_id, closure_from rows, max_interval_seconds, expected names)
# closure_of is always the single segment a: [70, 80]. Every segment here stays
# inside hour 1 (minutes 60-120) so the same cases can run under a period,
# which requires it; crossing a border is covered separately below.
PROXIMITY_SCENARIOS = [
    pytest.param([("a", 70, 80, "identical")], 0, ["identical"], id="identical"),
    pytest.param([("a", 73, 78, "inside")], 0, ["inside"], id="contained"),
    pytest.param([("a", 65, 90, "around")], 0, ["around"], id="containing"),
    pytest.param([("a", 65, 75, "overlaps_start")], 0, ["overlaps_start"], id="overlapping"),
    pytest.param([("a", 62, 70, "ends_at_start")], 0, ["ends_at_start"], id="touching_before"),
    pytest.param([("a", 80, 88, "starts_at_end")], 0, ["starts_at_end"], id="touching_after"),
    pytest.param([("a", 65, 69, "one_minute_before")], 0, [], id="just_before_no_allowance"),
    pytest.param([("a", 81, 85, "one_minute_after")], 0, [], id="just_after_no_allowance"),
    pytest.param([("a", 60, 65, "five_before")], 300, ["five_before"], id="gap_equals_allowance_before"),
    pytest.param([("a", 85, 90, "five_after")], 300, ["five_after"], id="gap_equals_allowance_after"),
    pytest.param([("a", 61, 64, "six_before")], 300, [], id="gap_over_allowance_before"),
    pytest.param([("a", 86, 90, "six_after")], 300, [], id="gap_over_allowance_after"),
    pytest.param([("a", 70, 70, "instant_inside")], 0, ["instant_inside"], id="instant_inside"),
    pytest.param([("a", 85, 85, "instant_near")], 300, ["instant_near"], id="instant_within_allowance"),
    pytest.param([("b", 70, 80, "other_key")], 3_000, [], id="other_key_never_matches"),
    pytest.param(
        [("a", 60, 65, "before"), ("a", 73, 74, "inside"),
         ("a", 85, 90, "after"), ("a", 110, 115, "far"), ("b", 70, 80, "wrong_key")],
        300, ["after", "before", "inside"],
        id="several_rows_at_once",
    ),
]


@pytest.mark.parametrize("frm, max_interval_seconds, expected", PROXIMITY_SCENARIOS)
def test_proximity(spark_session, frm, max_interval_seconds, expected):
    result = _closure(spark_session, [("a", 70, 80, "of")], frm, max_interval_seconds)
    assert _names(result) == expected


@pytest.mark.parametrize("frm, max_interval_seconds, expected", PROXIMITY_SCENARIOS)
def test_proximity_with_a_period(spark_session, frm, max_interval_seconds, expected):
    """The period must not change a single answer."""
    result = _closure(spark_session, [("a", 70, 80, "of")], frm, max_interval_seconds,
                      period=PeriodPartition.HOUR)
    assert _names(result) == expected


# --- the shape of the result -------------------------------------------------

def test_result_has_the_closure_from_columns(spark_session):
    """Nothing from closure_of, and no helper columns, come along."""
    result = _closure(spark_session, [("a", 60, 70, "of")], [("a", 60, 70, "from")])
    assert result.columns == CLOSURE_COLUMNS


def test_result_has_the_closure_from_columns_with_a_period(spark_session):
    result = _closure(spark_session, [("a", 60, 70, "of")], [("a", 60, 70, "from")],
                      period=PeriodPartition.HOUR)
    assert result.columns == CLOSURE_COLUMNS + ["et"]


def test_rows_are_returned_as_they_are(spark_session):
    result = _closure(spark_session, [("a", 60, 70, "of")], [("a", 62, 68, "from")])
    assert rows_of(result) == [("a", hhmm(62), hhmm(68), "from")]


def test_a_row_close_to_many_is_returned_once(spark_session):
    """The defining property of a semi join: no duplication."""
    of = [("a", 60, 70, "of1"), ("a", 61, 71, "of2"), ("a", 62, 72, "of3")]
    result = _closure(spark_session, of, [("a", 65, 66, "from")])
    assert _names(result) == ["from"]


def test_a_row_close_to_many_is_returned_once_with_a_period(spark_session):
    of = [("a", 55, 59, "of1"), ("a", 61, 65, "of2"), ("a", 121, 125, "of3")]
    result = _closure(spark_session, of, [("a", 60, 60, "from")], 300,
                      period=PeriodPartition.HOUR)
    assert _names(result) == ["from"]


def test_duplicate_from_rows_are_both_returned(spark_session):
    """Rows are filtered, not deduplicated."""
    result = _closure(spark_session, [("a", 60, 70, "of")],
                      [("a", 65, 66, "twin"), ("a", 65, 66, "twin")])
    assert _names(result) == ["twin", "twin"]


def test_an_empty_closure_of_selects_nothing(spark_session):
    result = _closure(spark_session, [], [("a", 60, 70, "from")])
    assert _names(result) == []


def test_an_empty_closure_from_gives_an_empty_result(spark_session):
    result = _closure(spark_session, [("a", 60, 70, "of")], [])
    assert result.columns == CLOSURE_COLUMNS
    assert _names(result) == []


# --- grouping columns --------------------------------------------------------

def test_several_group_by_columns(spark_session):
    columns = ["tenant", "key", "start_time", "end_time", "name"]
    of = frame(spark_session, columns, [("t1", "a", at(60), at(70), "of")])
    frm = frame(spark_session, columns, [
        ("t1", "a", at(65), at(66), "same_tenant_same_key"),
        ("t2", "a", at(65), at(66), "other_tenant"),
        ("t1", "b", at(65), at(66), "other_key"),
    ])
    result = compute_closure_by_time_proximity(
        of, frm, ["tenant", "key"], 300, "start_time", "end_time")
    assert _names(result) == ["same_tenant_same_key"]


def test_null_group_values_match_each_other(spark_session):
    of = frame(spark_session, CLOSURE_COLUMNS, [(None, at(60), at(70), "of")])
    frm = frame(spark_session, CLOSURE_COLUMNS, [
        (None, at(65), at(66), "also_null"),
        ("a", at(65), at(66), "not_null"),
    ])
    result = compute_closure_by_time_proximity(
        of, frm, ["key"], 300, "start_time", "end_time")
    assert _names(result) == ["also_null"]


# --- periods -----------------------------------------------------------------

def test_period_matches_across_a_border(spark_session):
    """The neighbouring period is searched, so a row just over the border is
    still found."""
    result = _closure(spark_session, [("a", 55, 59, "of")],
                      [("a", 61, 65, "next_hour"), ("a", 50, 54, "same_hour")],
                      300, period=PeriodPartition.HOUR)
    assert _names(result) == ["next_hour", "same_hour"]


def test_period_does_not_match_two_periods_away(spark_session):
    """Even with the largest allowance a period permits, a row two periods away
    is more than one whole period distant."""
    result = _closure(spark_session, [("a", 130, 140, "of")],
                      [("a", 0, 10, "two_hours_before")],
                      3_599, period=PeriodPartition.HOUR)
    assert _names(result) == []


def test_period_with_an_allowance_just_under_the_period(spark_session):
    """One second under the period is the largest allowance there is, and it
    still matches the unpartitioned answer."""
    of = [("a", 120, 130, "of")]
    frm = [("a", 61, 65, "an_hour_before"), ("a", 185, 189, "an_hour_after"),
           ("a", 0, 10, "long_before")]
    with_period = _closure(spark_session, of, frm, 3_599, period=PeriodPartition.HOUR)
    without = _closure(spark_session, of, frm, 3_599)
    assert _names(with_period) == _names(without) == ["an_hour_after", "an_hour_before"]


def test_day_period(spark_session):
    of = [("a", 24 * 60 + 30, 25 * 60, "of")]                    # day 2, 00:30-01:00
    frm = [("a", 23 * 60 + 40, 23 * 60 + 50, "late_on_day_1"),   # 40 min before
           ("a", 26 * 60, 27 * 60, "later_on_day_2"),            # 60 min after
           ("a", 5 * 60, 6 * 60, "early_on_day_1")]              # 18.5 h before
    result = _closure(spark_session, of, frm, 3_600, period=PeriodPartition.DAY)
    assert _names(result) == ["late_on_day_1", "later_on_day_2"]


CLOSURE_EQUIVALENCE_CASES = [
    pytest.param([("a", 10, 20, "of")], [("a", 15, 25, "f1")], id="both_inside_one_period"),
    pytest.param([("a", 55, 59, "of")], [("a", 61, 65, "f1")], id="across_a_border"),
    pytest.param([("a", 0, 5, "of")], [("a", 55, 59, "f1")], id="one_period_apart"),
    pytest.param([("a", 0, 5, "of")], [("a", 130, 140, "f1")], id="two_periods_apart"),
    pytest.param([("a", 59, 59, "of")], [("a", 60, 60, "f1")], id="instants_across_a_border"),
    pytest.param([("a", 10, 20, "of"), ("b", 70, 80, "of2")],
                 [("a", 25, 30, "f1"), ("b", 65, 66, "f2"), ("c", 10, 20, "f3")],
                 id="several_keys"),
    pytest.param([("a", 10, 20, "of")], [], id="no_from_rows"),
]


@pytest.mark.parametrize("of, frm", CLOSURE_EQUIVALENCE_CASES)
def test_period_gives_the_same_answer(spark_session, of, frm):
    """The period is an optimization only."""
    assert (_names(_closure(spark_session, of, frm, 300, period=PeriodPartition.HOUR))
            == _names(_closure(spark_session, of, frm, 300)))


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_period_gives_the_same_answer_on_random_input(spark_session, seed):
    rng = random.Random(seed)

    def build(count):
        rows = []
        for i in range(count):
            hour = rng.randint(0, 3)
            start = rng.randint(0, 59)
            # kept inside its hour, as the period contract requires
            end = min(59, start + rng.randint(0, 20))
            rows.append((rng.choice(["a", "b"]), hour * 60 + start, hour * 60 + end, "r%d" % i))
        return rows

    of, frm = build(15), build(30)
    assert (_names(_closure(spark_session, of, frm, 300, period=PeriodPartition.HOUR))
            == _names(_closure(spark_session, of, frm, 300)))


# --- validation --------------------------------------------------------------

def test_rejects_empty_group_by_columns(spark_session):
    with pytest.raises(ValueError, match="must not be empty"):
        _closure(spark_session, [("a", 0, 1, "of")], [("a", 0, 1, "f")], group_by_columns=[])


def test_rejects_a_negative_interval(spark_session):
    with pytest.raises(ValueError, match="must be >= 0"):
        _closure(spark_session, [("a", 0, 1, "of")], [("a", 0, 1, "f")], -1)


def test_reports_a_missing_column_by_frame(spark_session):
    of = _rows(spark_session, [("a", 0, 1, "of")])
    frm = _rows(spark_session, [("a", 0, 1, "f")])
    with pytest.raises(ValueError, match="columns not found in closure_of"):
        compute_closure_by_time_proximity(of.drop("key"), frm, ["key"], 300,
                                          "start_time", "end_time")
    with pytest.raises(ValueError, match="columns not found in closure_from"):
        compute_closure_by_time_proximity(of, frm.drop("end_time"), ["key"], 300,
                                          "start_time", "end_time")


def test_rejects_a_period_that_is_not_a_member(spark_session):
    of = _rows(spark_session, [("a", 0, 1, "of")])
    frm = _rows(spark_session, [("a", 0, 1, "f")])
    with pytest.raises(ValueError, match="must be a PeriodPartition"):
        compute_closure_by_time_proximity(of, frm, ["key"], 300,
                                          "start_time", "end_time", "HOUR")


def test_rejects_an_interval_as_long_as_the_period(spark_session):
    """One second under the period is allowed; the period itself is not."""
    of, frm = [("a", 0, 1, "of")], [("a", 0, 1, "f")]
    _closure(spark_session, of, frm, 3_599, period=PeriodPartition.HOUR)
    with pytest.raises(ValueError, match="must be shorter than one HOUR"):
        _closure(spark_session, of, frm, 3_600, period=PeriodPartition.HOUR)


def test_rejects_a_closure_of_segment_crossing_a_border(spark_session):
    with pytest.raises(ValueError, match="closure_of segment"):
        _closure(spark_session, [("a", 55, 65, "of")], [("a", 0, 1, "f")], 300,
                 period=PeriodPartition.HOUR)


def test_rejects_a_relevant_closure_from_segment_crossing_a_border(spark_session):
    """The offending row is in the hour next to closure_of, so it matters."""
    with pytest.raises(ValueError, match="closure_from segment"):
        _closure(spark_session, [("a", 10, 20, "of")], [("a", 55, 65, "f")], 300,
                 period=PeriodPartition.HOUR)


def test_ignores_an_irrelevant_closure_from_segment_crossing_a_border(spark_session):
    """A crossing row far from every closure_of period can never be selected,
    so it is not held to the single-period rule."""
    result = _closure(spark_session, [("a", 10, 20, "of")],
                      [("a", 15, 18, "near"), ("a", 295, 305, "far_and_crossing")],
                      300, period=PeriodPartition.HOUR)
    assert _names(result) == ["near"]


def test_a_crossing_closure_from_segment_is_fine_without_a_period(spark_session):
    """Without a period there is no single-period rule at all."""
    result = _closure(spark_session, [("a", 55, 65, "of")], [("a", 50, 70, "f")], 300)
    assert _names(result) == ["f"]


def test_period_column_is_not_required(spark_session):
    """The period is used through the timestamps; the partition column is not
    read, so a frame without it works."""
    of = _rows(spark_session, [("a", 10, 20, "of")])
    frm = _rows(spark_session, [("a", 25, 30, "f")])
    assert "et" not in of.columns
    result = compute_closure_by_time_proximity(
        of, frm, ["key"], 300, "start_time", "end_time", PeriodPartition.HOUR)
    assert _names(result) == ["f"]


# --- segments ending exactly on a border -------------------------------------

def test_a_segment_ending_on_a_border_is_inside_the_earlier_period(spark_session):
    """It fills hour 0 and occupies none of hour 1, which is also how
    split_on_period_borders cuts it, so it does not count as crossing."""
    result = _closure(spark_session, [("a", 30, 60, "of")], [("a", 40, 60, "from")], 300,
                      period=PeriodPartition.HOUR)
    assert _names(result) == ["from"]


def test_border_touching_rows_two_periods_apart_are_out_of_reach(spark_session):
    """closure_from ends exactly on the hour-1 border and closure_of starts
    exactly on the hour-2 border: two periods apart, and exactly one hour from
    each other. That is why the allowance has to stay under a period -- at the
    largest one allowed they are not close, so the three-period window loses
    nothing, and the partitioned and unpartitioned answers agree."""
    of = [("a", 120, 130, "of")]
    frm = [("a", 30, 60, "ends_on_the_hour_1_border")]
    with_period = _closure(spark_session, of, frm, 3_599, period=PeriodPartition.HOUR)
    without = _closure(spark_session, of, frm, 3_599)
    assert _names(with_period) == _names(without) == []


def test_a_stretched_result_can_be_fed_back_in(spark_session):
    """time_stretch splits its output on the borders, so its pieces end exactly
    on them; those pieces must pass the single-period validation done here."""
    from time_stretch import time_stretch

    stretched = time_stretch(
        segments(spark_session, [("a", 50, 58, 1), ("a", 60, 65, 2)],
                 with_period=PeriodPartition.HOUR),
        ["key"], [(F.sum, ["bytes"])], 300, "start_time", "end_time", PeriodPartition.HOUR)
    assert rows_of(stretched, ["start_time"]) == [
        ("a", hhmm(50), hhmm(60), 3, "2026010100"),
        ("a", hhmm(60), hhmm(65), 3, "2026010101"),
    ]

    result = compute_closure_by_time_proximity(
        stretched, stretched, ["key"], 300, "start_time", "end_time", PeriodPartition.HOUR)
    assert result.count() == 2


# --- the period is what makes the join an equality join ----------------------

def _join_keys(df):
    """The key lists of every join in the physical plan, as text."""
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        df.explain(mode="simple")
    return [line.strip() for line in buffer.getvalue().splitlines() if "Join" in line]


def test_the_period_becomes_a_join_key(spark_session):
    """The period condition cannot show up in the rows -- it only removes
    candidate pairs the time comparison would have rejected anyway. What it
    does change is the plan: the period index joins the hash keys, which is the
    whole point of passing a period. Without one, the times are left as a
    filter over every pair that shares a group.
    """
    of = _rows(spark_session, [("a", 10, 20, "of")])
    frm = _rows(spark_session, [("a", 15, 25, "f")])

    def keys(period):
        return " ".join(_join_keys(compute_closure_by_time_proximity(
            of, frm, ["key"], 300, "start_time", "end_time", period)))

    without, with_period = keys(None), keys(PeriodPartition.HOUR)
    # FLOOR(start_time / 3600) is period_index for an HOUR partition.
    assert "FLOOR" not in without, without
    assert "FLOOR" in with_period, with_period
    assert "Join" in without and "Join" in with_period
