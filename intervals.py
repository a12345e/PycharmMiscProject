from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.window import Window


def remove_contained_intervals(
        df: DataFrame,
        key_col: str = "key",
        start_col: str = "start_time",
        end_col: str = "end_time"
) -> DataFrame:
    """
    Remove rows whose [start, end] interval is fully contained in another
    interval with the same key.

    A row is removed when another row with the same key satisfies:
        other_start <= start  AND  end <= other_end
    Exact (key, start, end) duplicates keep a single copy.

    Returns:
    - DataFrame: the surviving rows, with the original columns.
    """
    # Sort each key's intervals by start asc, end desc: every earlier row in
    # the window starts at or before the current row, so the current row is
    # contained iff its end <= the max end seen among the earlier rows.
    preceding_window = (
        Window.partitionBy(key_col)
        .orderBy(F.col(start_col).asc(), F.col(end_col).desc())
        .rowsBetween(Window.unboundedPreceding, -1)
    )

    prev_max_end = F.max(F.col(end_col)).over(preceding_window)

    return (
        df.withColumn("_prev_max_end", prev_max_end)
        .filter(F.col("_prev_max_end").isNull() | (F.col(end_col) > F.col("_prev_max_end")))
        .drop("_prev_max_end")
    )

