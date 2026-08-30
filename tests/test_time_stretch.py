"""Unit tests for time_stretch.py. The code it shares with time_closure lives
in time_periods.py and is tested in test_time_periods.py; see conftest.py for
the fixtures and frame helpers."""

import random

import pytest
import pyspark.sql.functions as F

from tests.conftest import SEGMENT_COLUMNS, at, frame, hhmm, rows_of, segments
from time_periods import PeriodPartition, split_on_period_borders
from time_stretch import _assign_chain_ids, _validate, time_stretch


# --- _validate ---------------------------------------------------------------

def test_validate_returns_span_then_aggregations(spark_session):
    """The expressions rebuild the span first, then one column per aggregation."""
    df = segments(spark_session, [("a", 0, 1, 1)])
    exprs = _validate(df, ["key"], [(F.sum, ["bytes"]), (F.max, ["bytes", "start_time"])],
                      300, "start_time", "end_time", None)
    assert df.groupBy("key").agg(*exprs).columns == [
        "key", "start_time", "end_time", "bytes_sum", "bytes_max", "start_time_max",
    ]


def _validate_case(df, **overrides):
    kwargs = dict(group_by_columns=["key"], aggs=[(F.sum, ["bytes"])],
                  max_interval_seconds=300, start_time_column="start_time",
                  end_time_column="end_time", period=None)
    kwargs.update(overrides)
    return _validate(df, kwargs["group_by_columns"], kwargs["aggs"],
                     kwargs["max_interval_seconds"], kwargs["start_time_column"],
                     kwargs["end_time_column"], kwargs["period"])


@pytest.mark.parametrize(
    "overrides, message",
    [
        pytest.param({"group_by_columns": []}, "must not be empty", id="no_group_columns"),
        pytest.param({"max_interval_seconds": -1}, "must be >= 0", id="negative_interval"),
        pytest.param({"group_by_columns": ["nope"]}, "columns not found", id="unknown_group_column"),
        pytest.param({"start_time_column": "nope"}, "columns not found", id="unknown_start_column"),
        pytest.param({"end_time_column": "nope"}, "columns not found", id="unknown_end_column"),
        pytest.param({"aggs": [(F.sum, ["nope"])]}, "columns not found", id="unknown_agg_column"),
        pytest.param({"aggs": [(F.sum,)]}, "must be a (function, [columns]) pair", id="not_a_pair"),
        pytest.param({"aggs": [("sum", ["bytes"])]}, "is not callable", id="function_not_callable"),
        pytest.param({"aggs": [(F.sum, "bytes")]}, "must be a non-empty list", id="columns_as_string"),
        pytest.param({"aggs": [(F.sum, [])]}, "must be a non-empty list", id="columns_empty"),
        pytest.param({"aggs": [(F.sum, [1])]}, "must all be column names", id="column_not_a_name"),
        pytest.param({"aggs": [(F.sum, ["bytes"]), (F.sum, ["bytes"])]},
                     "produced more than once", id="duplicate_output_column"),
        pytest.param({"aggs": [(len, ["bytes"])]}, "failed on column", id="function_raises"),
        pytest.param({"aggs": [(lambda column: "not a column", ["bytes"])]},
                     "must be a Spark aggregation", id="function_returns_non_column"),
        pytest.param({"period": "HOUR"}, "must be a PeriodPartition", id="period_not_an_enum"),
        pytest.param({"period": PeriodPartition.HOUR}, "expects the partition column",
                     id="period_column_missing"),
    ],
)
def test_validate_rejects(spark_session, overrides, message):
    df = segments(spark_session, [("a", 0, 1, 1)])
    with pytest.raises(ValueError) as error:
        _validate_case(df, **overrides)
    assert message in str(error.value)


def test_validate_rejects_an_output_colliding_with_a_group_column(spark_session):
    """Grouping by "bytes_sum" while producing bytes_sum would shadow it."""
    df = frame(spark_session, ["bytes_sum", "start_time", "end_time", "bytes"],
               [("g", at(0), at(1), 1)])
    with pytest.raises(ValueError, match="produced more than once"):
        _validate(df, ["bytes_sum"], [(F.sum, ["bytes"])], 300,
                  "start_time", "end_time", None)


def test_validate_rejects_an_interval_longer_than_the_period(spark_session):
    """Partitioning cannot help when a chain may bridge a whole period."""
    df = segments(spark_session, [("a", 0, 1, 1)], with_period=PeriodPartition.HOUR)
    with pytest.raises(ValueError, match="must be shorter than one HOUR"):
        _validate_case(df, max_interval_seconds=3_600, period=PeriodPartition.HOUR)
    # one second under the period is still allowed
    _validate_case(df, max_interval_seconds=3_599, period=PeriodPartition.HOUR)


# --- _assign_chain_ids -------------------------------------------------------

def test_assign_chain_ids_numbers_chains_within_each_partition(spark_session):
    #   a: [0,10] [5,20] overlap -> chain 1;  [40,50] far -> chain 2
    #   b: [0,10] [12,20] gap 2min <= 5min -> chain 1
    df = segments(spark_session, [
        ("a", 0, 10, 1), ("a", 5, 20, 2), ("a", 40, 50, 3),
        ("b", 0, 10, 4), ("b", 12, 20, 5),
    ])
    chained = _assign_chain_ids(df, ["key"], "start_time", "end_time", 300, "chain")
    assert rows_of(chained.select("key", "bytes", "chain"), ["key", "bytes"]) == [
        ("a", 1, 1), ("a", 2, 1), ("a", 3, 2), ("b", 4, 1), ("b", 5, 1),
    ]


def test_assign_chain_ids_drops_its_helper_columns(spark_session):
    df = segments(spark_session, [("a", 0, 10, 1)])
    chained = _assign_chain_ids(df, ["key"], "start_time", "end_time", 300, "chain")
    assert chained.columns == SEGMENT_COLUMNS + ["chain"]


def test_assign_chain_ids_uses_the_running_max_end(spark_session):
    """[0,100] swallows the rows after it even though they start later: the
    chain continues while the *largest* end seen so far is close enough."""
    df = segments(spark_session, [("a", 0, 100, 1), ("a", 10, 20, 2), ("a", 103, 110, 3)])
    chained = _assign_chain_ids(df, ["key"], "start_time", "end_time", 300, "chain")
    assert {row["chain"] for row in chained.collect()} == {1}


# --- time_stretch, no period -------------------------------------------------

AGGS = [(F.sum, ["bytes"]), (F.max, ["bytes"]), (F.count, ["bytes"])]


def _stretch(spark, rows, max_interval_seconds=300, period=None, aggs=AGGS):
    df = segments(spark, rows, with_period=period)
    return time_stretch(df, ["key"], aggs, max_interval_seconds,
                        "start_time", "end_time", period)


# (test_id, segments, max_interval_seconds, expected (key, start, end, sum, max, count))
STRETCH_SCENARIOS = [
    pytest.param(
        [("a", 0, 10, 1)], 300,
        [("a", 0, 10, 1, 1, 1)],
        id="single_segment_passes_through",
    ),
    pytest.param(
        # overlapping segments merge, whatever the gap allowance
        [("a", 0, 10, 1), ("a", 5, 20, 2)], 0,
        [("a", 0, 20, 3, 2, 2)],
        id="overlapping_merge",
    ),
    pytest.param(
        # touching exactly: the gap is 0, so 0 is enough to merge
        [("a", 0, 10, 1), ("a", 10, 20, 2)], 0,
        [("a", 0, 20, 3, 2, 2)],
        id="touching_merge_at_zero_allowance",
    ),
    pytest.param(
        # one minute apart with no allowance stays apart
        [("a", 0, 10, 1), ("a", 11, 20, 2)], 0,
        [("a", 0, 10, 1, 1, 1), ("a", 11, 20, 2, 2, 1)],
        id="apart_at_zero_allowance",
    ),
    pytest.param(
        # the gap is exactly max_interval_seconds -> merged (5 minutes)
        [("a", 0, 10, 1), ("a", 15, 20, 2)], 300,
        [("a", 0, 20, 3, 2, 2)],
        id="gap_equal_to_the_allowance_merges",
    ),
    pytest.param(
        # one second more than the allowance -> not merged
        [("a", 0, 10, 1), ("a", 15, 20, 2)], 299,
        [("a", 0, 10, 1, 1, 1), ("a", 15, 20, 2, 2, 1)],
        id="gap_over_the_allowance_splits",
    ),
    pytest.param(
        # a segment fully inside another
        [("a", 0, 100, 1), ("a", 10, 20, 2)], 0,
        [("a", 0, 100, 3, 2, 2)],
        id="contained_segment",
    ),
    pytest.param(
        # the chain continues from the running max end, not from the last row
        [("a", 0, 100, 1), ("a", 10, 20, 2), ("a", 104, 110, 4)], 300,
        [("a", 0, 110, 7, 4, 3)],
        id="chain_follows_the_running_max_end",
    ),
    pytest.param(
        # duplicated rows collapse into one segment but still count twice
        [("a", 0, 10, 1), ("a", 0, 10, 1)], 0,
        [("a", 0, 10, 2, 1, 2)],
        id="duplicate_rows",
    ),
    pytest.param(
        # a zero-length segment inside a chain
        [("a", 0, 10, 1), ("a", 10, 10, 2)], 0,
        [("a", 0, 10, 3, 2, 2)],
        id="zero_length_inside_a_chain",
    ),
    pytest.param(
        # keys never merge with each other, even at identical times
        [("a", 0, 10, 1), ("b", 0, 10, 2)], 300,
        [("a", 0, 10, 1, 1, 1), ("b", 0, 10, 2, 2, 1)],
        id="keys_are_independent",
    ),
    pytest.param(
        # input order must not matter: the same chain, backwards
        [("a", 40, 50, 3), ("a", 5, 20, 2), ("a", 0, 10, 1)], 1_800,
        [("a", 0, 50, 6, 3, 3)],
        id="input_order_does_not_matter",
    ),
    pytest.param(
        # three chains separated by long gaps
        [("a", 0, 10, 1), ("a", 60, 70, 2), ("a", 120, 130, 4)], 300,
        [("a", 0, 10, 1, 1, 1), ("a", 60, 70, 2, 2, 1), ("a", 120, 130, 4, 4, 1)],
        id="several_chains",
    ),
]


@pytest.mark.parametrize("rows, max_interval_seconds, expected", STRETCH_SCENARIOS)
def test_time_stretch_without_a_period(spark_session, rows, max_interval_seconds, expected):
    result = _stretch(spark_session, rows, max_interval_seconds)
    assert rows_of(result, ["key", "start_time"]) == [
        (key, hhmm(start), hhmm(end), total, largest, count)
        for key, start, end, total, largest, count in expected
    ]


def test_time_stretch_output_columns(spark_session):
    result = _stretch(spark_session, [("a", 0, 10, 1)],
                      aggs=[(F.sum, ["bytes"]), (F.min, ["bytes", "start_time"])])
    assert result.columns == ["key", "start_time", "end_time",
                              "bytes_sum", "bytes_min", "start_time_min"]


def test_time_stretch_groups_by_several_columns(spark_session):
    df = frame(spark_session, ["tenant", "key", "start_time", "end_time", "bytes"], [
        ("t1", "a", at(0), at(10), 1),
        ("t1", "a", at(12), at(20), 2),
        ("t2", "a", at(0), at(10), 4),   # same key, other tenant -> separate
    ])
    result = time_stretch(df, ["tenant", "key"], [(F.sum, ["bytes"])], 300,
                          "start_time", "end_time")
    assert rows_of(result, ["tenant", "start_time"]) == [
        ("t1", "a", hhmm(0), hhmm(20), 3),
        ("t2", "a", hhmm(0), hhmm(10), 4),
    ]


def test_time_stretch_keeps_a_null_group_value(spark_session):
    df = frame(spark_session, SEGMENT_COLUMNS,
               [(None, at(0), at(10), 1), (None, at(12), at(20), 2), ("a", at(0), at(10), 4)])
    result = time_stretch(df, ["key"], [(F.sum, ["bytes"])], 300, "start_time", "end_time")
    assert sorted(rows_of(result), key=lambda row: (row[0] or "")) == [
        (None, hhmm(0), hhmm(20), 3),
        ("a", hhmm(0), hhmm(10), 4),
    ]


def test_time_stretch_rejects_a_non_aggregate_function(spark_session):
    """Not a ValueError: Spark's analyzer catches it while the plan is built."""
    from pyspark.errors import AnalysisException

    df = segments(spark_session, [("a", 0, 10, 1)])
    with pytest.raises(AnalysisException):
        time_stretch(df, ["key"], [(F.upper, ["bytes"])], 300, "start_time", "end_time")


def _merge_reference(rows, max_interval_minutes):
    """Plain-Python interval merging, per key."""
    merged = {}
    for key, start, end, value in sorted(rows, key=lambda s: (s[0], s[1], s[2])):
        chains = merged.setdefault(key, [])
        if chains and start - chains[-1][1] <= max_interval_minutes:
            chains[-1][1] = max(chains[-1][1], end)
            chains[-1][2] += value
        else:
            chains.append([start, end, value])
    return sorted((key, start, end, total)
                  for key, chains in merged.items() for start, end, total in chains)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_time_stretch_matches_a_plain_python_merge(spark_session, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(50):
        key = rng.choice(["a", "b", "c"])
        start = rng.randint(0, 240)
        rows.append((key, start, start + rng.randint(0, 30), rng.randint(1, 9)))

    result = _stretch(spark_session, rows, 300, aggs=[(F.sum, ["bytes"])])
    assert rows_of(result, ["key", "start_time"]) == [
        (key, hhmm(start), hhmm(end), total)
        for key, start, end, total in _merge_reference(rows, 5)
    ]


# --- time_stretch, with a period ---------------------------------------------

def test_time_stretch_stitches_and_splits_on_the_border(spark_session):
    #   a: two segments inside hour 0            -> one row, hour 0
    #   b: 50-58 and 60-65, gap 2min <= 5min     -> stretched over the border,
    #                                               then split back into two rows
    #   c: 10-20 and 60-70, gap 40min > 5min     -> two separate rows
    rows = [
        ("a", 10, 20, 1), ("a", 22, 40, 2),
        ("b", 50, 58, 4), ("b", 60, 65, 8),
        ("c", 10, 20, 16), ("c", 60, 70, 32),
    ]
    result = _stretch(spark_session, rows, 300, period=PeriodPartition.HOUR)
    assert rows_of(result, ["key", "start_time"]) == [
        ("a", hhmm(10), hhmm(40), 3, 2, 2, "2026010100"),
        # one chain, one row per period, each carrying the whole chain's values
        ("b", hhmm(50), hhmm(60), 12, 8, 2, "2026010100"),
        ("b", hhmm(60), hhmm(65), 12, 8, 2, "2026010101"),
        ("c", hhmm(10), hhmm(20), 16, 16, 1, "2026010100"),
        ("c", hhmm(60), hhmm(70), 32, 32, 1, "2026010101"),
    ]


def test_time_stretch_with_a_period_does_not_stitch_across_keys(spark_session):
    """Two keys with segments hugging the same border stay separate."""
    rows = [("a", 55, 59, 1), ("b", 61, 65, 2)]
    result = _stretch(spark_session, rows, 300, period=PeriodPartition.HOUR)
    assert rows_of(result, ["key"]) == [
        ("a", hhmm(55), hhmm(59), 1, 1, 1, "2026010100"),
        ("b", hhmm(61), hhmm(65), 2, 2, 1, "2026010101"),
    ]


def test_time_stretch_with_a_period_spans_three_periods(spark_session):
    """A chain of within-hour segments, each close to the next, becomes one
    stretched chain and comes back as one row per hour it covers."""
    rows = [("a", 50, 59, 1), ("a", 61, 119, 2), ("a", 121, 130, 4)]
    result = _stretch(spark_session, rows, 300, period=PeriodPartition.HOUR)
    assert rows_of(result, ["start_time"]) == [
        ("a", hhmm(50), hhmm(60), 7, 4, 3, "2026010100"),
        ("a", hhmm(60), hhmm(120), 7, 4, 3, "2026010101"),
        ("a", hhmm(120), hhmm(130), 7, 4, 3, "2026010102"),
    ]


def test_time_stretch_with_a_period_rejects_a_crossing_segment(spark_session):
    df = segments(spark_session, [("a", 55, 65, 1)], with_period=PeriodPartition.HOUR)
    with pytest.raises(ValueError, match="spans more than one HOUR period"):
        time_stretch(df, ["key"], AGGS, 300, "start_time", "end_time", PeriodPartition.HOUR)


def test_time_stretch_with_a_period_requires_the_partition_column(spark_session):
    df = segments(spark_session, [("a", 0, 10, 1)])   # no "et" column
    with pytest.raises(ValueError, match="expects the partition column"):
        time_stretch(df, ["key"], AGGS, 300, "start_time", "end_time", PeriodPartition.HOUR)


PARTITION_EQUIVALENCE_CASES = [
    pytest.param([("a", 10, 20, 1), ("a", 22, 40, 2)], id="all_inside_one_period"),
    pytest.param([("a", 50, 59, 1), ("a", 61, 70, 2)], id="one_chain_over_a_border"),
    pytest.param([("a", 50, 59, 1), ("a", 61, 119, 2), ("a", 121, 130, 4)],
                 id="one_chain_over_three_borders"),
    pytest.param([("a", 0, 59, 1), ("a", 60, 119, 2)], id="segments_filling_whole_periods"),
    pytest.param([("a", 10, 20, 1), ("b", 55, 59, 2), ("b", 61, 65, 4), ("c", 130, 140, 8)],
                 id="several_keys"),
    pytest.param([("a", 59, 59, 1), ("a", 60, 60, 2)], id="instants_on_both_sides_of_a_border"),
]


@pytest.mark.parametrize("rows", PARTITION_EQUIVALENCE_CASES)
def test_partitioned_equals_unpartitioned_then_split(spark_session, rows):
    """The partitioning is an optimization: stretching per period and stitching
    must give exactly what stretching the whole timeline and splitting gives."""
    partitioned = _stretch(spark_session, rows, 300, period=PeriodPartition.HOUR)
    plain = _stretch(spark_session, rows, 300)
    split = split_on_period_borders(plain, "start_time", "end_time", PeriodPartition.HOUR)
    assert rows_of(partitioned, ["key", "start_time"]) == rows_of(split, ["key", "start_time"])


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_partitioned_equals_unpartitioned_on_random_input(spark_session, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(40):
        key = rng.choice(["a", "b"])
        hour = rng.randint(0, 3)
        start = rng.randint(0, 59)
        # kept inside its hour, as the period contract requires
        end = min(59, start + rng.randint(0, 20))
        rows.append((key, hour * 60 + start, hour * 60 + end, rng.randint(1, 9)))

    partitioned = _stretch(spark_session, rows, 300, period=PeriodPartition.HOUR)
    plain = _stretch(spark_session, rows, 300)
    split = split_on_period_borders(plain, "start_time", "end_time", PeriodPartition.HOUR)
    assert rows_of(partitioned, ["key", "start_time"]) == rows_of(split, ["key", "start_time"])
