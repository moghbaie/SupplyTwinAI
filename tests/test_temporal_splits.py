import polars as pl
import pytest

from sctwin.splits import assign_roles, make_manifest, read_manifest, write_manifest

KW = dict(fit_days=14, val_days=6, holdout_store_frac=0.25, holdout_product_frac=0.2, seed=7)


@pytest.fixture
def manifest(train_test):
    return make_manifest(train_test[0], **KW)


def test_fit_val_partition_dates(train_test, manifest):
    tr = assign_roles(train_test[0], manifest)
    fit = tr.filter(pl.col("role") == "fit")["dt"].str.to_date()
    val = tr.filter(pl.col("role") == "val")["dt"].str.to_date()
    assert fit.max() < val.min()
    assert fit.n_unique() == 14 and val.n_unique() == 6
    assert tr.height == train_test[0].height


def test_eval_strictly_after_val(train_test, manifest):
    te = train_test[1]
    assert (
        te["dt"].str.to_date().min()
        > pl.Series([manifest["temporal"]["val"]["end"]]).str.to_date()[0]
    )


def test_wrong_day_count_rejected(train_test):
    with pytest.raises(ValueError, match="distinct dates"):
        make_manifest(train_test[0], **{**KW, "fit_days": 10})


def test_group_holdout_rule(train_test, manifest):
    tr = assign_roles(train_test[0], manifest)
    g = manifest["group_holdout"]
    assert len(g["stores"]) == 1 and len(g["products"]) == 1
    ho = tr.filter(pl.col("holdout"))
    assert (ho["store_id"].is_in(g["stores"]) | ho["product_id"].is_in(g["products"])).all()
    ins = tr.filter(~pl.col("holdout"))
    assert not ins["store_id"].is_in(g["stores"]).any()
    assert not ins["product_id"].is_in(g["products"]).any()
    c = manifest["series_counts"]
    assert (
        c["in_sample"] + c["holdout_store_only"] + c["holdout_product_only"] + c["holdout_both"]
        == c["total"]
    )


def test_manifest_deterministic_and_order_independent(train_test):
    a = make_manifest(train_test[0], **KW)
    b = make_manifest(train_test[0].sample(fraction=1.0, shuffle=True, seed=3), **KW)
    assert a == b


def test_manifest_write_once(tmp_path, manifest):
    p = tmp_path / "m.json"
    write_manifest(manifest, p)
    write_manifest(manifest, p)  # identical: no-op
    changed = {**manifest, "seed": 999}
    with pytest.raises(FileExistsError):
        write_manifest(changed, p)
    assert read_manifest(p) == manifest


def test_manifest_tamper_detected(tmp_path, manifest):
    p = tmp_path / "m.json"
    write_manifest(manifest, p)
    p.write_text(p.read_text().replace('"seed": 7', '"seed": 8'))
    with pytest.raises(ValueError, match="digest"):
        read_manifest(p)
