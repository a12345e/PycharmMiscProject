import pytest
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, TimestampType,
)
from clock_align import find_optimal_session_offset

import sys, os


@pytest.fixture(scope="session")
def spark_session():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    """Fixture to manage a local Spark Session lifecycle during tests."""
    spark = (
        SparkSession.builder
        .master("local[1]")  # Limit threads per worker to 1 or 2 max
        .appName(f"pytest-spark")
        # Core Speed Optimizations:
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")  # DEFAULT IS 200 (Huge bottleneck)
        .config("spark.default.parallelism", "1")
        .config("spark.ui.enabled", "false")  # Disable Web UI overhead
        # Isolation for Parallel Workers:
        .config("spark.sql.warehouse.dir", f"/tmp/spark-warehouse-aa")
        .config("spark.driver.extraJavaOptions", "-Djava.net.preferIPv4Stack=true")
        .getOrCreate())
    yield spark
    spark.stop()


def test_spark_session_is_utc(spark_session):
    """Guard: the session must run in UTC so timestamp handling is deterministic."""
    assert spark_session.conf.get("spark.sql.session.timeZone") == "UTC"
    tz = spark_session.sql("SELECT current_timezone() AS tz").collect()[0]["tz"]
    assert tz == "UTC"


# Shared session timeline used by every scenario:
#   session_start = 12:00:00, session_end = 13:00:00 (window length = 3600s)
BASE_TIME = datetime(2026, 1, 1, 12, 0, 0)
SESSION_START = BASE_TIME
SESSION_END = BASE_TIME + timedelta(hours=1)

# Explicit schemas so the number columns are IntegerType (Python ints would
# otherwise be inferred as LongType).
SESSION_SCHEMA = StructType([
    StructField("session_id", StringType()),
    StructField("session_upper_num", IntegerType()),
    StructField("session_lower_num", IntegerType()),
    StructField("session_start_t", TimestampType()),
    StructField("session_end_t", TimestampType()),
    StructField("session_name", StringType()),
])
EVENT_SCHEMA = StructType([
    StructField("event_id", StringType()),
    StructField("event_num", IntegerType()),
    StructField("event_group", StringType()),
    StructField("event_t", TimestampType()),
])


# Reusable event-offset lists. Offsets are measured from SESSION_END (events are
# built as SESSION_END + offset), so an offset in seconds equals the event's
# min_offset, and min_offset + 3600 (the window length) is its max_offset.

# Peak at +3550: 8-event cluster at +3550 plus 2 stragglers opening slightly
# earlier (+3541/+3542); all share the clipped right edge +3600.
TEST1_OFFSETS = [timedelta(minutes=59, seconds=1),
                 timedelta(minutes=59, seconds=2)] + [timedelta(minutes=59, seconds=10)] * 8

# A cluster of negative-side events whose match windows all extend across 0:
# the 8-event cluster sits at -3550 (interval [-3550, 50]) and the 2 stragglers
# at -3559/-3558 (intervals [-3559, 41] / [-3558, 42]). The peak coverage of 10
# holds over the plateau [-3550, 41], which straddles 0 -> the minimal-|offset|
# point with maximal matches is 0.
TEST2_OFFSETS = [timedelta(seconds=-3559),
                 timedelta(seconds=-3558)] + [timedelta(seconds=-3550)] * 8

# 20 events evenly spread across the session window, inclusive of both ends:
# i=0 -> -3600 (event at SESSION_START), i=19 -> 0 (event at SESSION_END).
# Every in-session event's interval contains offset 0, and the SESSION_END event
# makes 0 itself a boundary point, so the max-coverage point collapses onto 0.
SPREAD_OFFSETS = [timedelta(seconds=round(i * 3600 / 19) - 3600) for i in range(20)]


# (test_id, event_offsets_from_session_end, max_offset_seconds, expected_offset, expected_matches)
SCENARIOS = [
    pytest.param(
        # All 10 events first overlap at offset 3550 (window [12:59:10, 13:59:10]);
        # every offset in [3550, 3600] yields 10 matches, and the function breaks
        # the tie by minimizing |offset|, so 3550 is optimal.
        TEST1_OFFSETS,
        3600, 3550, 10,
        id="maximal_offset_edge_case",
    ),
    pytest.param(
        # The peak-coverage plateau [-3550, 41] straddles 0, so the minimal-|offset|
        # offset with the maximal 10 matches is 0.
        TEST2_OFFSETS,
        3600, 0, 10,
        id="negative_cluster_resolves_to_zero",
    ),
    pytest.param(
        # test-1 events (peak +3550) + the negative-side test-2 cluster + 20 events
        # spread across the session. At offset 0 the 20 spread events and the 10
        # test-2 events all match (their intervals contain 0) while the 10 test-1
        # events do not -> 30 matches, the strict global max, so offset 0 wins.
        TEST1_OFFSETS + TEST2_OFFSETS + SPREAD_OFFSETS,
        3600, 0, 30,
        id="combined_with_spread_chooses_zero",
    ),
    pytest.param(
        # A single event exactly at session_end matches with no shift -> offset 0.
        [timedelta(0)],
        3600, 0, 1,
        id="single_event_at_session_end",
    ),
    pytest.param(
        # Event sits 2h past session_end, beyond the 3600s offset window, so it can
        # never match -> no matching pairs -> (0, 0).
        [timedelta(hours=2)],
        3600, 0, 0,
        id="event_outside_offset_window",
    ),
]
@pytest.mark.parametrize(
    "event_offsets, max_offset_seconds, expected_offset, expected_matches",
    SCENARIOS,
)
def test_find_optimal_session_offset(spark_session, event_offsets, max_offset_seconds,
                                     expected_offset, expected_matches):
    # Arrange
    session_data = [(
        "matching_token_123",  # session exact match field
        100,  # session_upper_num
        1,    # session_lower_num
        SESSION_START,
        SESSION_END,
        "Test Session",
    )]
    session_df = spark_session.createDataFrame(session_data, SESSION_SCHEMA)
    session_df.printSchema()
    event_data = [
        ("matching_token_123", 50, "GroupA", SESSION_END + offset) for offset in event_offsets
    ]
    event_df = spark_session.createDataFrame(event_data, EVENT_SCHEMA)

    print("session_df")
    session_df.show(100,truncate=False)
    print("event_df")
    event_df.show(100,truncate=False)

    # Act
    optimal_offset, max_matches = find_optimal_session_offset(
        session_df=session_df,
        event_df=event_df,
        max_offset_seconds=max_offset_seconds,
        session_start_time_col="session_start_t",
        session_end_time_col="session_end_t",
        event_time_col="event_t",
        session_number_lower_bound_col="session_lower_num",
        session_number_upper_bound_col="session_upper_num",
        event_number_col="event_num",
        session_exact_match_col="session_id",      # Mapped to 'session_id'
        event_exact_match_col="event_id",  # Mapped to 'event_id'
    )

    # Assert
    assert optimal_offset == expected_offset
    assert max_matches == expected_matches


# --- Multiple sessions, a single event, all sessions "away" from the event ---
#
# One event joins against 10 sessions. Each (session, event) pair yields an offset
# interval [event_t - s_end, event_t - s_start]; the result is the offset of maximal
# overlap, and within that plateau the offset closest to 0.
#
# 5 "peak" sessions share an end-time 3600s from the event, so their intervals share
# a left edge at -direction*3600; the 5 overlap over a plateau where matches peak at
# 5. For sessions earlier than the event the plateau is positive [3600, 4200] -> the
# closest-to-0 offset is its left edge +3600; for sessions later than the event the
# plateau is negative [-3600, -3000] -> the closest-to-0 offset is its right edge
# -3000. The other 5 "decoy" sessions sit much further away (|offset| ~ 5000-5860),
# are short, and don't overlap each other, so each contributes a single match.

EVENT_TIME = BASE_TIME                       # 2026-01-01 12:00:00
MULTI_SESSION_MAX_OFFSET = 7200              # 2h window; nothing here is clipped
PEAK_DURATIONS = [600, 1200, 1800, 2400, 3000]


def _build_away_sessions(direction):
    """Build 10 sessions all on one side of the single event.

    direction = +1 -> sessions later than the event   (best offset -3000)
    direction = -1 -> sessions earlier than the event  (best offset +3600)
    """
    sessions = []

    # 5 peak sessions: shared end 3600s from the event, varying durations.
    peak_end = EVENT_TIME + timedelta(seconds=direction * 3600)
    for dur in PEAK_DURATIONS:
        peak_start = peak_end - timedelta(seconds=dur)
        start_t, end_t = min(peak_start, peak_end), max(peak_start, peak_end)
        sessions.append(("matching_token_123", 100, 1, start_t, end_t, "peak"))

    # 5 decoy sessions: short, far away, non-overlapping -> one match each.
    for i in range(5):
        far = 5000 + 200 * i
        a = EVENT_TIME + timedelta(seconds=direction * far)
        b = EVENT_TIME + timedelta(seconds=direction * (far + 60))
        start_t, end_t = min(a, b), max(a, b)
        sessions.append(("matching_token_123", 100, 1, start_t, end_t, "decoy"))

    return sessions


@pytest.mark.parametrize(
    "direction, expected_offset",
    [
        pytest.param(1, -3000, id="ten_sessions_after_event"),
        pytest.param(-1, 3600, id="ten_sessions_before_event"),
    ],
)
def test_multiple_sessions_single_event(spark_session, direction, expected_offset):
    # Arrange
    session_df = spark_session.createDataFrame(_build_away_sessions(direction), SESSION_SCHEMA)
    event_df = spark_session.createDataFrame(
        [("matching_token_123", 50, "GroupA", EVENT_TIME)], EVENT_SCHEMA
    )

    print("session_df")
    session_df.show(100, truncate=False)
    print("event_df")
    event_df.show(100, truncate=False)

    # Act
    optimal_offset, max_matches = find_optimal_session_offset(
        session_df=session_df,
        event_df=event_df,
        max_offset_seconds=MULTI_SESSION_MAX_OFFSET,
        session_start_time_col="session_start_t",
        session_end_time_col="session_end_t",
        event_time_col="event_t",
        session_number_lower_bound_col="session_lower_num",
        session_number_upper_bound_col="session_upper_num",
        event_number_col="event_num",
        session_exact_match_col="session_id",
        event_exact_match_col="event_id",
    )

    # Assert: 5 of the 10 sessions overlap at the chosen offset.
    assert max_matches == 5
    assert optimal_offset == expected_offset


# --- 10 exact-match values, each with two event rows and two session rows ---
#
# For each of the 10 tokens there are two events two hours apart, and each event
# has a matching session whose start is 15 minutes *after* that event. The
# sessions are instantaneous (start == end) so the offset that aligns an event
# with its session is exactly event_t - session_start = -900s; with a non-zero
# duration the reported offset (the left edge of the overlap plateau) would drift
# by the duration, so zero-duration sessions are what make the best offset land
# exactly on "15 minutes".
#
# At offset -900 every event meets its own session: 10 tokens x 2 events = 20
# matches -- the global maximum. The cross pairs (event_1 vs session_for_e2, and
# vice-versa) only ever line up at -8100s and +6300s with 10 matches each, so
# -900 wins.

def test_ten_tokens_best_offset_is_fifteen_minutes(spark_session):
    # Arrange
    two_hours = timedelta(hours=2)
    fifteen_minutes = timedelta(minutes=15)

    session_rows = []
    event_rows = []
    for k in range(10):
        token = f"token_{k}"
        base = BASE_TIME + timedelta(hours=k)   # distinct, non-overlapping per token
        e1 = base
        e2 = base + two_hours                    # the two events are two hours apart
        s1 = e1 + fifteen_minutes                # matching session start, 15 min above e1
        s2 = e2 + fifteen_minutes                # matching session start, 15 min above e2

        event_rows += [
            (token, 50, "GroupA", e1),
            (token, 50, "GroupA", e2),
        ]
        # Instantaneous sessions (start == end) anchored at the 15-minute mark.
        session_rows += [
            (token, 100, 1, s1, s1, "session_for_e1"),
            (token, 100, 1, s2, s2, "session_for_e2"),
        ]

    session_df = spark_session.createDataFrame(session_rows, SESSION_SCHEMA)
    event_df = spark_session.createDataFrame(event_rows, EVENT_SCHEMA)

    print("session_df")
    session_df.show(100, truncate=False)
    print("event_df")
    event_df.show(100, truncate=False)

    # Act
    optimal_offset, max_matches = find_optimal_session_offset(
        session_df=session_df,
        event_df=event_df,
        max_offset_seconds=10800,   # 3h window: nothing here is clipped
        session_start_time_col="session_start_t",
        session_end_time_col="session_end_t",
        event_time_col="event_t",
        session_number_lower_bound_col="session_lower_num",
        session_number_upper_bound_col="session_upper_num",
        event_number_col="event_num",
        session_exact_match_col="session_id",
        event_exact_match_col="event_id",
    )

    # Assert: shifting sessions back 15 minutes aligns all 20 event/session pairs.
    assert optimal_offset == -900          # -15 minutes
    assert max_matches == 20


# --- Pairs that can only meet beyond +/-max_offset_seconds are ignored ---
#
# 4 events match within the window (at offset 1800) while 6 events sit two hours
# past session_end, needing an offset of 7200s > the 3600s limit. The 6 are
# dropped by the range filter; if they were counted they would dominate (6 > 4),
# so the result proves they are ignored.

def test_events_unmatchable_within_max_offset_are_ignored(spark_session):
    # Arrange
    session_df = spark_session.createDataFrame(
        [("match", 100, 1, SESSION_START, SESSION_END, "S")], SESSION_SCHEMA
    )
    in_range = [("match", 50, "G", SESSION_END + timedelta(seconds=1800))] * 4
    out_of_range = [("match", 50, "G", SESSION_END + timedelta(seconds=7200))] * 6
    event_df = spark_session.createDataFrame(in_range + out_of_range, EVENT_SCHEMA)

    print("session_df")
    session_df.show(100, truncate=False)
    print("event_df")
    event_df.show(100, truncate=False)

    # Act
    optimal_offset, max_matches = find_optimal_session_offset(
        session_df=session_df,
        event_df=event_df,
        max_offset_seconds=3600,
        session_start_time_col="session_start_t",
        session_end_time_col="session_end_t",
        event_time_col="event_t",
        session_number_lower_bound_col="session_lower_num",
        session_number_upper_bound_col="session_upper_num",
        event_number_col="event_num",
        session_exact_match_col="session_id",
        event_exact_match_col="event_id",
    )

    # Assert: only the 4 reachable events count; the 6 unreachable ones are ignored.
    assert optimal_offset == 1800
    assert max_matches == 4


# --- Wrong exact-match key or out-of-range event number are ignored ---
#
# Only the 3 events with the right token AND an event_num inside [1, 100] should
# join. The other 6 (wrong token, or number above/below the bounds) are excluded
# by the join condition; if they were not, all 9 share the same time and would
# count as 9.

def test_wrong_exact_match_or_out_of_number_range_are_ignored(spark_session):
    # Arrange
    session_df = spark_session.createDataFrame(
        [("match", 100, 1, SESSION_START, SESSION_END, "S")], SESSION_SCHEMA
    )
    t = SESSION_END + timedelta(seconds=600)
    valid = [("match", 50, "G", t)] * 3        # right token, event_num within [1, 100]
    wrong_token = [("other", 50, "G", t)] * 2  # exact-match field differs
    num_too_high = [("match", 500, "G", t)] * 2  # event_num above upper bound (100)
    num_too_low = [("match", 0, "G", t)] * 2     # event_num below lower bound (1)
    event_df = spark_session.createDataFrame(
        valid + wrong_token + num_too_high + num_too_low, EVENT_SCHEMA
    )

    print("session_df")
    session_df.show(100, truncate=False)
    print("event_df")
    event_df.show(100, truncate=False)

    # Act
    optimal_offset, max_matches = find_optimal_session_offset(
        session_df=session_df,
        event_df=event_df,
        max_offset_seconds=3600,
        session_start_time_col="session_start_t",
        session_end_time_col="session_end_t",
        event_time_col="event_t",
        session_number_lower_bound_col="session_lower_num",
        session_number_upper_bound_col="session_upper_num",
        event_number_col="event_num",
        session_exact_match_col="session_id",
        event_exact_match_col="event_id",
    )

    # Assert: only the 3 valid events join; the 6 mismatching events are ignored.
    assert optimal_offset == 600
    assert max_matches == 3


# --- Simplest case: one session, one event already inside it -> offset 0 ---
#
# The event (10:00) sits inside the session window [10:00, 11:00], so it stays
# matched for every offset in [-3600, 0]; the minimal-|offset| offset with the
# maximal (single) match is 0 -- no shift is needed.

def test_single_session_event_inside_gives_zero_offset(spark_session):
    # Arrange
    session_start = datetime(2023, 1, 1, 10, 0, 0)
    session_end = datetime(2023, 1, 1, 11, 0, 0)
    event_time = datetime(2023, 1, 1, 10, 0, 0)

    session_df = spark_session.createDataFrame(
        [("match", 100, 1, session_start, session_end, "S")], SESSION_SCHEMA
    )
    event_df = spark_session.createDataFrame(
        [("match", 50, "G", event_time)], EVENT_SCHEMA
    )

    print("session_df")
    session_df.show(truncate=False)
    print("event_df")
    event_df.show(truncate=False)

    # Act
    optimal_offset, max_matches = find_optimal_session_offset(
        session_df=session_df,
        event_df=event_df,
        max_offset_seconds=3600,
        session_start_time_col="session_start_t",
        session_end_time_col="session_end_t",
        event_time_col="event_t",
        session_number_lower_bound_col="session_lower_num",
        session_number_upper_bound_col="session_upper_num",
        event_number_col="event_num",
        session_exact_match_col="session_id",
        event_exact_match_col="event_id",
    )

    # Assert: no shift needed.
    assert optimal_offset == 0
    assert max_matches == 1


# --- Every event falls inside every session it joins -> offset 0 -------------
#
# When an event time lies within a session window (start <= event_t <= end), the
# pair's offset interval [event_t - end, event_t - start] satisfies
# event_t - end <= 0 <= event_t - start, i.e. it spans 0. If this holds for every
# joined (session, event) pair, then offset 0 matches ALL pairs at once -- the
# largest possible count -- so the optimal offset is 0 and max_matches equals the
# total number of joined pairs. Each scenario below is one such "all events inside
# all sessions" configuration; all must resolve to offset 0 with no shift.

HOUR = timedelta(hours=1)


def _inside_session(token, start, end):
    """A session row (number range [1, 100]) spanning [start, end]."""
    return (token, 100, 1, start, end, "S")


def _inside_event(token, t):
    """An event row whose number 50 sits inside every session's [1, 100] range."""
    return (token, 50, "GroupA", t)


INSIDE_SCENARIOS = [
    pytest.param(
        # One session [12:00, 13:00] with three events strictly inside it.
        [_inside_session("tok", SESSION_START, SESSION_END)],
        [_inside_event("tok", SESSION_START + timedelta(minutes=15)),
         _inside_event("tok", SESSION_START + timedelta(minutes=30)),
         _inside_event("tok", SESSION_START + timedelta(minutes=45))],
        3,
        id="single_session_three_events_inside",
    ),
    pytest.param(
        # Three nested sessions, all containing all three events. The tightest
        # session is [11:30, 13:30] and every event sits within it (and thus
        # within the wider two as well) -> 3 sessions x 3 events = 9 pairs at 0.
        [_inside_session("tok", BASE_TIME - HOUR, BASE_TIME + 2 * HOUR),
         _inside_session("tok", BASE_TIME - timedelta(minutes=30),
                         BASE_TIME + timedelta(minutes=90)),
         _inside_session("tok", BASE_TIME - 2 * HOUR, BASE_TIME + 3 * HOUR)],
        [_inside_event("tok", BASE_TIME),
         _inside_event("tok", BASE_TIME + timedelta(minutes=30)),
         _inside_event("tok", BASE_TIME + HOUR)],
        9,
        id="three_nested_sessions_all_contain_all_events",
    ),
    pytest.param(
        # Events sitting exactly on the inclusive session boundaries (start and
        # end) still match at offset 0, alongside one in the middle.
        [_inside_session("tok", SESSION_START, SESSION_END)],
        [_inside_event("tok", SESSION_START),                       # at start
         _inside_event("tok", SESSION_START + timedelta(minutes=30)),  # middle
         _inside_event("tok", SESSION_END)],                        # at end
        3,
        id="events_on_inclusive_session_boundaries",
    ),
    pytest.param(
        # Two exact-match tokens; within each token every event lies inside its
        # session. Cross-token pairs never join, so at offset 0 all 2 + 2 pairs
        # match -> 4, the global max.
        [_inside_session("tok_a", SESSION_START, SESSION_END),
         _inside_session("tok_b", BASE_TIME + 2 * HOUR, BASE_TIME + 3 * HOUR)],
        [_inside_event("tok_a", SESSION_START + timedelta(minutes=20)),
         _inside_event("tok_a", SESSION_START + timedelta(minutes=40)),
         _inside_event("tok_b", BASE_TIME + timedelta(minutes=140)),
         _inside_event("tok_b", BASE_TIME + timedelta(minutes=160))],
        4,
        id="two_tokens_events_inside_their_sessions",
    ),
]


@pytest.mark.parametrize("session_rows, event_rows, expected_matches", INSIDE_SCENARIOS)
def test_events_inside_all_sessions_gives_zero_offset(
    spark_session, session_rows, event_rows, expected_matches
):
    # Arrange
    session_df = spark_session.createDataFrame(session_rows, SESSION_SCHEMA)
    event_df = spark_session.createDataFrame(event_rows, EVENT_SCHEMA)

    print("session_df")
    session_df.show(100, truncate=False)
    print("event_df")
    event_df.show(100, truncate=False)

    # Act
    optimal_offset, max_matches = find_optimal_session_offset(
        session_df=session_df,
        event_df=event_df,
        max_offset_seconds=7200,   # generous window: the offset-0 plateau is never clipped
        session_start_time_col="session_start_t",
        session_end_time_col="session_end_t",
        event_time_col="event_t",
        session_number_lower_bound_col="session_lower_num",
        session_number_upper_bound_col="session_upper_num",
        event_number_col="event_num",
        session_exact_match_col="session_id",
        event_exact_match_col="event_id",
    )

    # Assert: events already inside their sessions need no shift.
    assert optimal_offset == 0
    assert max_matches == expected_matches


# --- Very simple one-session scenarios over a 10:00-11:00 window -------------
#
# Single session [10:00, 11:00], number range [1000, 2000], max_offset 3600s.
# Every event uses number 1500 (inside the range) so the join always succeeds
# and only the event TIMES drive the result.
#
#   1. event at 10:00 (session start)  -> already inside -> offset 0, 1 match.
#   2. event at 11:00 (session end)    -> already inside (inclusive) -> 0, 1.
#   3. five events at 11:00, 10:01, 10:59, 09:59, 11:01. Each event's offset
#      interval is [event_t - 11:00, event_t - 10:00] (clipped to +/-3600):
#         11:00 -> [   0, 3600]
#         10:01 -> [-3540,  60]
#         10:59 -> [ -60, 3540]
#         09:59 -> [-3600, -60]   (clipped)
#         11:01 -> [  60, 3600]   (clipped)
#      The unique max-overlap point is +60, covered by 11:00/10:01/10:59/11:01
#      -> offset 60, 4 matches.

SIMPLE_SESSION_START = datetime(2026, 1, 1, 10, 0, 0)
SIMPLE_SESSION_END = datetime(2026, 1, 1, 11, 0, 0)


def _minutes(h, m):
    """A timestamp on the test day at hour h, minute m."""
    return datetime(2026, 1, 1, h, m, 0)


SIMPLE_SCENARIOS = [
    pytest.param(
        [_minutes(10, 0)],
        0, 1,
        id="event_at_session_start",
    ),
    pytest.param(
        [_minutes(11, 0)],
        0, 1,
        id="event_at_session_end",
    ),
    pytest.param(
        [_minutes(11, 1), _minutes(10, 0)
         ],
        0, 1,
        id="two events one inside another with positive offset required peek ad not offset",
    ),
    pytest.param(
        [_minutes(11, 0), _minutes(9, 59)
         ],
        0, 1,
        id="two events one inside another with negative offset required peek ad not offset",
    ),
    pytest.param(
        [_minutes(10, 30), _minutes(9, 59)
         ],
        -60, 2,
        id="one event in the session middle and another below but not far below",
    ),
    pytest.param(
        [_minutes(10, 30), _minutes(11, 1)
         ],
        60, 2,
        id="one event in the session middle and another upper but not far upper",
    ),
    pytest.param(
            [_minutes(10, 30), _minutes(10, 31),_minutes(10, 29),
                    _minutes(12, 30), _minutes(12, 31),_minutes(12, 22)],
        0, 3,
        id="three outsiders require positive offset and three cannot do with the minimal offset for the outsiders so offsset=0",
    ),
    pytest.param(
            [_minutes(10, 30), _minutes(10, 31),_minutes(10, 29),
                    _minutes(8, 30), _minutes(8, 31),_minutes(8, 22)],
        0, 3,
        id="three outsiders require negative offset and three cannot do with the minimal offset for the outsiders so offsset=0",
    ),
    pytest.param(
            [_minutes(10, 30), _minutes(11, 30)],
        1800, 2,
        id="one inside in the middle and another outside such offset 30 is good for both",
    ),
    pytest.param(
            [_minutes(10, 30), _minutes(11, 31)],
        0, 1,
        id="one is above but too far for offset to work",
    ),
    pytest.param(
        [_minutes(11, 0), _minutes(10, 1), _minutes(10, 59),
         _minutes(9, 59), _minutes(11, 1)],
        60, 4,
        id="five_events_peak_at_plus_60",
    ),
]
@pytest.mark.parametrize("event_times, expected_offset, expected_matches", SIMPLE_SCENARIOS)
def test_simple_one_session_scenarios(spark_session, event_times,
                                      expected_offset, expected_matches):
    # Arrange: one session [10:00, 11:00], number range [1000, 2000].
    session_df = spark_session.createDataFrame(
        [("match", 2000, 1000, SIMPLE_SESSION_START, SIMPLE_SESSION_END, "S")],
        SESSION_SCHEMA,
    )
    # Every event uses number 1500, which is inside [1000, 2000].
    event_df = spark_session.createDataFrame(
        [("match", 1500, "G", t) for t in event_times], EVENT_SCHEMA
    )

    print("session_df")
    session_df.show(truncate=False)
    print("event_df")
    event_df.show(truncate=False)

    # Act
    optimal_offset, max_matches = find_optimal_session_offset(
        session_df=session_df,
        event_df=event_df,
        max_offset_seconds=3600,
        session_start_time_col="session_start_t",
        session_end_time_col="session_end_t",
        event_time_col="event_t",
        session_number_lower_bound_col="session_lower_num",
        session_number_upper_bound_col="session_upper_num",
        event_number_col="event_num",
        session_exact_match_col="session_id",
        event_exact_match_col="event_id",
    )

    # Assert
    assert optimal_offset == expected_offset
    assert max_matches == expected_matches