"""Shared fixtures and helpers for the time_periods / time_stretch /
time_closure tests.

Note on building DataFrames: the frames here are built with SQL VALUES rather
than spark.createDataFrame(). They are equivalent, but createDataFrame() pushes
a Python list through the RDD pickler, which crashes on Python 3.12+ with
pyspark 3.5 -- SQL VALUES stays JVM-side and works on every version.

Note on timestamps: collect() converts a timestamp to the driver's local time,
which hides what Spark actually stored. The helpers here therefore render
timestamps with F.date_format, i.e. in the session timezone (UTC).
"""

import os
import sys
from datetime import datetime, timedelta

import pytest
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from time_periods import PeriodPartition


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


BASE_TIME = datetime(2026, 1, 1, 0, 0, 0)


def at(minutes):
    """A timestamp `minutes` minutes after BASE_TIME."""
    return BASE_TIME + timedelta(minutes=minutes)


def hhmm(minutes):
    """How `at(minutes)` is rendered by the assertions below."""
    return at(minutes).strftime("%d %H:%M")


def literal(value):
    """A Python value as a typed SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, datetime):
        return "TIMESTAMP '%s'" % value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return "%dL" % value
    if isinstance(value, float):
        return "CAST(%r AS DOUBLE)" % value
    return "'%s'" % value


def frame(spark, columns, rows):
    """A DataFrame with `columns`, built JVM-side from `rows` of Python values."""
    values = ", ".join("(%s)" % ", ".join(literal(v) for v in row) for row in rows)
    return spark.sql("SELECT * FROM VALUES %s AS tbl(%s)" % (values, ", ".join(columns)))


def rows_of(df, order=None, time_format="dd HH:mm"):
    """The rows as tuples, with timestamps rendered by Spark in UTC."""
    if order:
        df = df.orderBy(*order)
    types = dict(df.dtypes)
    projected = [
        F.date_format(F.col(c), time_format).alias(c) if types[c] == "timestamp" else F.col(c)
        for c in df.columns
    ]
    return [tuple(row) for row in df.select(*projected).collect()]


# Segments as (key, start_minutes, end_minutes, bytes), the shape most tests use.
SEGMENT_COLUMNS = ["key", "start_time", "end_time", "bytes"]

_PERIOD_STRFTIME = {"yyyyMMddHH": "%Y%m%d%H", "yyyyMMdd": "%Y%m%d",
                    "yyyyMM": "%Y%m", "yyyy": "%Y"}


def segments(spark, rows, with_period=None, columns=SEGMENT_COLUMNS):
    """A segment frame; with_period also adds the partition column."""
    columns = list(columns)
    rows = [(key, at(start), at(end), value) for key, start, end, value in rows]
    if with_period is not None:
        columns.append(with_period.column)
        fmt = _PERIOD_STRFTIME[with_period.fmt]
        rows = [row + (row[1].strftime(fmt),) for row in rows]
    return frame(spark, columns, rows)


class SixHours:
    """A size > 1 period. PeriodPartition has no such member yet, and the enum
    cannot be extended, so this stands in to exercise the size arithmetic."""

    fmt = "yyyyMMddHH"
    size = 6
    unit = "hour"
    min_seconds = 6 * 3_600
    column = PeriodPartition.COLUMN
    name = "SIX_HOURS"

    @property
    def interval(self):
        return F.expr("INTERVAL 6 HOUR")


SIX_HOURS = SixHours()
