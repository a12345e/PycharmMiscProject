from datetime import datetime, timedelta
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, IntegerType, TimestampType


# --- The Updated Function ---

def _find_optimal_session_offset_prepare(
        session_df: DataFrame,
        event_df: DataFrame,
        session_start_time_col: str,
        session_end_time_col: str,
        event_time_col: str,
        session_number_lower_bound_col: str,
        session_number_upper_bound_col: str,
        event_number_col: str,
        session_exact_match_col: str,
        event_exact_match_col: str
) -> tuple:
    """
    Validate the input schemas, project down to the columns the optimization uses,
    and drop rows where any required field is null.

    Raises:
    - ValueError: if a required column is missing.
    - TypeError: if a required column has an unexpected dtype.

    Returns:
    - tuple: (cleaned_session_df, cleaned_event_df)
    """

    # Validate schemas, project to the columns we use, and drop rows with nulls
    session_required_cols = [
        session_exact_match_col,
        session_number_lower_bound_col,
        session_number_upper_bound_col,
        session_start_time_col,
        session_end_time_col,
    ]
    event_required_cols = [
        event_exact_match_col,
        event_number_col,
        event_time_col,
    ]

    missing_session_cols = [c for c in session_required_cols if c not in session_df.columns]
    if missing_session_cols:
        raise ValueError(f"session_df is missing required columns: {missing_session_cols}")

    missing_event_cols = [c for c in event_required_cols if c not in event_df.columns]
    if missing_event_cols:
        raise ValueError(f"event_df is missing required columns: {missing_event_cols}")

    # Validate the dtype of every column we use: exact-match -> string,
    # number bounds -> integer, time columns -> timestamp.
    type_expectations = [
        (session_df, session_exact_match_col, StringType, "string"),
        (event_df, event_exact_match_col, StringType, "string"),
        (session_df, session_number_lower_bound_col, IntegerType, "integer"),
        (session_df, session_number_upper_bound_col, IntegerType, "integer"),
        (event_df, event_number_col, IntegerType, "integer"),
        (session_df, session_start_time_col, TimestampType, "timestamp"),
        (session_df, session_end_time_col, TimestampType, "timestamp"),
        (event_df, event_time_col, TimestampType, "timestamp"),
    ]
    for df, col, expected_type, expected_label in type_expectations:
        actual_type = df.schema[col].dataType
        if not isinstance(actual_type, expected_type):
            raise TypeError(
                f"Column '{col}' must be of type {expected_label}, "
                f"but found {actual_type.simpleString()}"
            )

    # Keep only the columns we use, then remove rows where any required field is null
    session_df = session_df.select(*session_required_cols).dropna(subset=session_required_cols)
    event_df = event_df.select(*event_required_cols).dropna(subset=event_required_cols)

    return session_df, event_df


def find_optimal_session_offset(
        session_df: DataFrame,
        event_df: DataFrame,
        max_offset_seconds: int,
        session_start_time_col: str,
        session_end_time_col: str,
        event_time_col: str,
        session_number_lower_bound_col: str,
        session_number_upper_bound_col: str,
        event_number_col: str,
        session_exact_match_col: str,
        event_exact_match_col: str
) -> tuple:
    """
    Finds the optimal time offset within [-max_offset_seconds, max_offset_seconds]
    that maximizes the number of matching events based on explicit column mappings.
    Ties are broken by minimizing the absolute value of the offset.

    Returns:
    - tuple: (optimal_offset_seconds, max_events_matched)
    """

    # 0. Validate, project, and clean the inputs.
    session_df, event_df = _find_optimal_session_offset_prepare(
        session_df,
        event_df,
        session_start_time_col,
        session_end_time_col,
        event_time_col,
        session_number_lower_bound_col,
        session_number_upper_bound_col,
        event_number_col,
        session_exact_match_col,
        event_exact_match_col,
    )

    # 1. Join tables on non-time constraints dynamically using distinct exact match columns
    matched_pairs = session_df.join(
        event_df,
        (session_df[session_exact_match_col] == event_df[event_exact_match_col]) &
        (event_df[event_number_col] >= session_df[session_number_lower_bound_col]) &
        (event_df[event_number_col] <= session_df[session_number_upper_bound_col])
    )

    # 2. Calculate the raw offset bounds required for each event to match its session
    pairs_with_offsets = matched_pairs.withColumn(
        "min_offset_sec", (F.col(event_time_col).cast("long") - F.col(session_end_time_col).cast("long"))
    ).withColumn(
        "max_offset_sec", (F.col(event_time_col).cast("long") - F.col(session_start_time_col).cast("long"))
    )

    # 3. Filter pairs that can overlap within the max offset window, then clip boundaries
    lower_limit = -max_offset_seconds
    upper_limit = max_offset_seconds

    valid_ranges = pairs_with_offsets.filter(
        (F.col("max_offset_sec") >= lower_limit) & (F.col("min_offset_sec") <= upper_limit)
    ).withColumn(
        "valid_min", F.when(F.col("min_offset_sec") < lower_limit, lower_limit).otherwise(F.col("min_offset_sec"))
    ).withColumn(
        "valid_max", F.when(F.col("max_offset_sec") > upper_limit, upper_limit).otherwise(F.col("max_offset_sec"))
    )

    # 4. Flatten intervals into boundary points (+1 when entering, -1 when exiting)
    starts = valid_ranges.select(F.col("valid_min").alias("offset_point"), F.lit(1).alias("point_value"))
    ends = valid_ranges.select((F.col("valid_max") + 1).alias("offset_point"), F.lit(-1).alias("point_value"))

    valid_ranges.show(truncate=False)

    timeline = starts.union(ends)

    # 5. Aggregate changes at identical offset points
    aggregated_timeline = timeline.groupBy("offset_point").agg(F.sum("point_value").alias("net_change"))

    # 6. Calculate, for every boundary point, the coverage that holds over the
    #    half-open plateau [offset_point, next_offset_point) and where it ends.
    order_window = Window.orderBy("offset_point")
    running_window = order_window.rowsBetween(Window.unboundedPreceding, Window.currentRow)

    timeline_with_counts = aggregated_timeline.withColumn(
        "total_matches", F.sum("net_change").over(running_window)
    ).withColumn(
        "next_offset_point", F.lead("offset_point").over(order_window)
    )

    # 7. Every offset inside a plateau yields the same number of matches, so within
    #    the winning plateau [offset_point, next_offset_point - 1] we pick the offset
    #    with the smallest absolute value:
    #      - 0 if the plateau straddles zero,
    #      - the left edge (offset_point) if the plateau is entirely positive,
    #      - the right edge (next_offset_point - 1) if the plateau is entirely negative.
    timeline_with_best = timeline_with_counts.withColumn(
        "best_offset",
        F.when((F.col("offset_point") <= 0) & (F.col("next_offset_point") >= 1), F.lit(0))
         .when(F.col("offset_point") > 0, F.col("offset_point"))
         .otherwise(F.col("next_offset_point") - 1)
    )

    # Rank by most matches, then minimal |offset| (ties broken deterministically).
    ranked = timeline_with_best.orderBy(
        F.desc("total_matches"),
        F.abs(F.col("best_offset")).asc(),
        F.col("best_offset").desc(), #positive offset better then negative
    )
    print('ranked')
    ranked.show(truncate=False)
    optimal_row = ranked.first()

    if optimal_row is None:
        print("No matching pairs found -> chosen offset=0, count=0")
        return 0, 0

    # Top 10 offsets by match count (ties broken by minimal absolute offset)
    top_offsets = ranked.limit(10).collect()

    print("Top 10 offsets (best_offset, total_matches):")
    for row in top_offsets:
        print(f"  offset={row['best_offset']}, count={row['total_matches']}")
    print(f"Chosen -> offset={optimal_row['best_offset']}, count={optimal_row['total_matches']}")

    return optimal_row["best_offset"], optimal_row["total_matches"]