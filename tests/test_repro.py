import json

from sctwin.repro import config_hash, seed_everything, write_run_manifest


def test_config_hash_key_order_independent():
    assert config_hash({"a": 1, "b": {"c": 2, "d": 3}}) == config_hash(
        {"b": {"d": 3, "c": 2}, "a": 1}
    )
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_seed_determinism():
    a = seed_everything(5).normal(size=5)
    b = seed_everything(5).normal(size=5)
    assert (a == b).all()


def test_run_manifest(tmp_path):
    p = write_run_manifest(tmp_path, {"seed": 1}, stage="test")
    m = json.loads(p.read_text())
    assert m["stage"] == "test" and m["environment"]["python"]
