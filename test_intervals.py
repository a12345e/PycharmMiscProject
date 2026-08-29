import os
import random
import sys
from datetime import datetime, timedelta

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, TimestampType,
)

from intervals import remove_contained_intervals


@pytest.fixture(scope="session")
def spark_session():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    """Fixture to manage a local Spark Session lifecycle during tests."""
    spark = (
        SparkSession.builder
        .master("local[1]")
        .appName("pytest-spark")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.default.parallelism", "1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse-aa")
        .config("spark.driver.extraJavaOptions", "-Djava.net.preferIPv4Stack=true")
        .getOrCreate())
    yield spark
    spark.stop()


# Intervals are written as plain integers (minutes from BASE_TIME) so the
# scenarios stay readable; they are converted to timestamps before the call.
BASE_TIME = datetime(2026, 1, 1, 0, 0, 0)

SCHEMA = StructType([
    StructField("key", StringType()),
    StructField("start_time", TimestampType()),
    StructField("end_time", TimestampType()),
    StructField("label", StringType()),
])


def _t(minutes):
    """A timestamp `minutes` minutes after BASE_TIME."""
    return BASE_TIME + timedelta(minutes=minutes)


def _to_rows(intervals):
    """(key, start, end) integer triples -> DataFrame rows with a label column."""
    return [
        (key, _t(start), _t(end), f"{key}:{start}-{end}")
        for key, start, end in intervals
    ]


def _collect_triples(df):
    """Result rows back as sorted (key, start_minutes, end_minutes) triples."""
    return sorted(
        (
            row["key"],
            int((row["start_time"] - BASE_TIME).total_seconds() // 60),
            int((row["end_time"] - BASE_TIME).total_seconds() // 60),
        )
        for row in df.collect()
    )


def _brute_force(intervals):
    """Reference implementation: keep the distinct intervals that no *other*
    distinct interval of the same key contains."""
    distinct = set(intervals)
    survivors = [
        (key, start, end)
        for (key, start, end) in distinct
        if not any(
            k == key and (s, e) != (start, end) and s <= start and end <= e
            for (k, s, e) in distinct
        )
    ]
    return sorted(survivors)


# (test_id, input intervals, expected surviving intervals)
SCENARIOS = [
    pytest.param(
        [("a", 0, 10)],
        [("a", 0, 10)],
        id="single_row_is_kept",
    ),
    pytest.param(
        # [2, 5] sits strictly inside [0, 10].
        [("a", 0, 10), ("a", 2, 5)],
        [("a", 0, 10)],
        id="strictly_contained_is_removed",
    ),
    pytest.param(
        # Same (key, start, end) three times -> exactly one copy survives.
        [("a", 0, 10), ("a", 0, 10), ("a", 0, 10)],
        [("a", 0, 10)],
        id="exact_duplicates_collapse_to_one",
    ),
    pytest.param(
        # Shared start: the shorter one is contained (start <= start, end <= end).
        [("a", 0, 10), ("a", 0, 4)],
        [("a", 0, 10)],
        id="shared_start_keeps_the_longer",
    ),
    pytest.param(
        # Shared end: [5, 10] is contained in [0, 10].
        [("a", 0, 10), ("a", 5, 10)],
        [("a", 0, 10)],
        id="shared_end_keeps_the_longer",
    ),
    pytest.param(
        # Overlap without containment -> both survive.
        [("a", 0, 5), ("a", 3, 8)],
        [("a", 0, 5), ("a", 3, 8)],
        id="partial_overlap_keeps_both",
    ),
    pytest.param(
        # Disjoint intervals are never contained in one another.
        [("a", 0, 2), ("a", 3, 4)],
        [("a", 0, 2), ("a", 3, 4)],
        id="disjoint_keeps_both",
    ),
    pytest.param(
        # Touching at a single point is still not containment.
        [("a", 0, 5), ("a", 5, 9)],
        [("a", 0, 5), ("a", 5, 9)],
        id="touching_endpoints_keeps_both",
    ),
    pytest.param(
        # Nested chain: only the outermost survives, transitively.
        [("a", 0, 100), ("a", 10, 90), ("a", 20, 80), ("a", 40, 50)],
        [("a", 0, 100)],
        id="nested_chain_keeps_only_outermost",
    ),
    pytest.param(
        # Input order must not matter: the same chain, innermost first.
        [("a", 40, 50), ("a", 20, 80), ("a", 10, 90), ("a", 0, 100)],
        [("a", 0, 100)],
        id="nested_chain_reversed_input_order",
    ),
    pytest.param(
        # A zero-length interval inside another is contained and removed.
        [("a", 0, 10), ("a", 5, 5)],
        [("a", 0, 10)],
        id="zero_length_inside_is_removed",
    ),
    pytest.param(
        # A zero-length interval with nothing covering it survives.
        [("a", 0, 10), ("a", 20, 20)],
        [("a", 0, 10), ("a", 20, 20)],
        id="zero_length_outside_is_kept",
    ),
    pytest.param(
        # Containment is per key: identical intervals under different keys
        # never remove each other.
        [("a", 0, 10), ("b", 2, 5), ("b", 0, 10), ("a", 2, 5)],
        [("a", 0, 10), ("b", 0, 10)],
        id="containment_is_scoped_per_key",
    ),
    pytest.param(
        # Two keys with independent shapes: 'a' collapses, 'b' keeps a staircase.
        [("a", 0, 100), ("a", 10, 20),
         ("b", 0, 5), ("b", 4, 9), ("b", 8, 13)],
        [("a", 0, 100), ("b", 0, 5), ("b", 4, 9), ("b", 8, 13)],
        id="two_keys_independent_results",
    ),
    pytest.param(
        # A staircase where each interval extends past the previous one: the
        # running max end grows every row, so nothing is dropped.
        [("a", 0, 5), ("a", 2, 9), ("a", 6, 12), ("a", 11, 20)],
        [("a", 0, 5), ("a", 2, 9), ("a", 6, 12), ("a", 11, 20)],
        id="staircase_keeps_everything",
    ),
    pytest.param(
        # A later, wider interval must remove an earlier-starting? No: [3, 20]
        # starts after [0, 10] and ends after it, so both survive; [4, 9] is
        # covered by both and goes.
        [("a", 0, 10), ("a", 3, 20), ("a", 4, 9)],
        [("a", 0, 10), ("a", 3, 20)],
        id="contained_by_a_later_starting_interval",
    ),
    pytest.param(
        # Duplicates of a contained interval are removed along with the original.
        [("a", 0, 10), ("a", 2, 5), ("a", 2, 5), ("a", 0, 10)],
        [("a", 0, 10)],
        id="duplicated_contained_intervals_all_removed",
    ),
    pytest.param(
        # Every key has exactly one row -> nothing can be contained.
        [("a", 0, 1), ("b", 0, 1), ("c", 0, 1)],
        [("a", 0, 1), ("b", 0, 1), ("c", 0, 1)],
        id="one_row_per_key",
    ),
]


@pytest.mark.parametrize("intervals, expected", SCENARIOS)
def test_remove_contained_intervals(spark_session, intervals, expected):
    # Arrange
    df = spark_session.createDataFrame(_to_rows(intervals), SCHEMA)

    print("input")
    df.show(100, truncate=False)

    # Act
    result = remove_contained_intervals(df)

    print("result")
    result.show(100, truncate=False)

    # Assert
    assert _collect_triples(result) == sorted(expected)
    # The scenarios are also a check on the reference implementation used by
    # the randomized test below.
    assert _brute_force(intervals) == sorted(expected)


def test_empty_input_returns_empty(spark_session):
    # Arrange
    df = spark_session.createDataFrame([], SCHEMA)

    # Act
    result = remove_contained_intervals(df)

    # Assert
    assert result.count() == 0
    assert result.columns == df.columns


def test_original_columns_are_preserved(spark_session):
    """The helper column used internally must not leak into the output, and the
    surviving rows must carry their other columns unchanged."""
    # Arrange
    df = spark_session.createDataFrame(
        _to_rows([("a", 0, 10), ("a", 2, 5), ("b", 7, 8)]), SCHEMA
    )

    # Act
    result = remove_contained_intervals(df)

    # Assert
    assert result.columns == ["key", "start_time", "end_time", "label"]
    assert sorted(row["label"] for row in result.collect()) == ["a:0-10", "b:7-8"]


def test_custom_column_names(spark_session):
    # Arrange
    schema = StructType([
        StructField("device", StringType()),
        StructField("from_t", TimestampType()),
        StructField("to_t", TimestampType()),
    ])
    rows = [
        ("d1", _t(0), _t(10)),
        ("d1", _t(3), _t(7)),    # contained
        ("d1", _t(9), _t(15)),   # overlapping, survives
        ("d2", _t(3), _t(7)),    # different device, survives
    ]
    df = spark_session.createDataFrame(rows, schema)

    # Act
    result = remove_contained_intervals(
        df, key_col="device", start_col="from_t", end_col="to_t"
    )

    # Assert
    assert sorted(
        (row["device"], row["from_t"], row["to_t"]) for row in result.collect()
    ) == [
        ("d1", _t(0), _t(10)),
        ("d1", _t(9), _t(15)),
        ("d2", _t(3), _t(7)),
    ]


# --- Randomized cross-check against the brute-force reference ---------------
#
# Small integer ranges keep duplicates, shared edges and nesting frequent, which
# is exactly where the window-function trick is easy to get wrong.

@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_matches_brute_force_on_random_input(spark_session, seed):
    # Arrange
    rng = random.Random(seed)
    intervals = []
    for _ in range(60):
        key = rng.choice(["a", "b", "c"])
        start = rng.randint(0, 12)
        end = start + rng.randint(0, 8)
        intervals.append((key, start, end))

    df = spark_session.createDataFrame(_to_rows(intervals), SCHEMA)

    # Act
    result = remove_contained_intervals(df)

    # Assert
    assert _collect_triples(result) == _brute_force(intervals)
