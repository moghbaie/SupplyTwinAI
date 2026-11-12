import numpy as np
import polars as pl
import pytest

from sctwin.data import hourly_matrix, to_hourly_long, validate_schema


def test_schema_ok_on_fixture(frame):
    rep = validate_schema(frame)
    assert rep.ok, rep


def test_schema_detects_missing_and_wrong_type(frame):
    bad = frame.drop("discount").with_columns(pl.col("sale_amount").cast(pl.Utf8))
    rep = validate_schema(bad)
    assert "discount" in rep.missing
    assert rep.wrong_type["sale_amount"] == ("float", "str")
    assert not rep.ok


def test_hourly_matrix_shape_and_values(frame):
    m = hourly_matrix(frame, "hours_sale")
    assert m.shape == (frame.height, 24)
    np.testing.assert_allclose(m.sum(axis=1), frame["sale_amount"].to_numpy())


def test_hourly_matrix_rejects_wrong_length(frame):
    bad = frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.col("hours_sale").list.head(23))
        .otherwise(pl.col("hours_sale"))
        .alias("hours_sale")
    )
    with pytest.raises(ValueError, match="expected 24"):
        hourly_matrix(bad, "hours_sale")


def test_hourly_long_preserves_totals(frame):
    long = to_hourly_long(frame).collect()
    assert long.height == frame.height * 24
    assert set(long["hour"].unique().to_list()) == set(range(24))
    np.testing.assert_allclose(long["sale"].sum(), frame["sale_amount"].sum())
    assert long["stockout"].sum() == frame["hours_stock_status"].list.sum().sum()
    # hour index aligns with array position
    first = long.filter(
        (pl.col("store_id") == 100) & (pl.col("product_id") == 500) & (pl.col("dt") == "2024-03-01")
    ).sort("hour")
    np.testing.assert_allclose(first["sale"].to_numpy(), frame["hours_sale"][0].to_numpy())
