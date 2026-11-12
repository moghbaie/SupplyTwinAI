"""Rolling-origin backtesting (CLAUDE.md Rule 1), in a configurable unit.

At origin T (end of unit T):

* **training examples**: feature time t with target window (t + L, t + L + H]; included only if
  the full window ends at or before T, i.e. ``t <= T - L - H``. Window: expanding (from
  ``min_history``) or sliding (the last ``window_length`` feature times). ``train_stride``
  thins feature times (a subset of past examples; anchored at the latest allowed t).
* **prediction**: features as of T, targets (T + L, T + L + H].

:func:`check_split` asserts both conditions on every fold at run time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class BacktestSpec:
    unit: str
    lead: int  # L
    horizon: int  # H
    origins: tuple[int, ...]  # fold origins T (unit indices)
    window: str = "sliding"  # "sliding" | "expanding"
    window_length: int = 0  # W, number of feature times for a sliding window
    min_history: int = 0  # earliest feature time allowed
    train_stride: int = 1

    def __post_init__(self) -> None:
        if self.window not in ("sliding", "expanding"):
            raise ValueError("window must be 'sliding' or 'expanding'")
        if self.window == "sliding" and self.window_length < 1:
            raise ValueError("sliding window needs window_length >= 1")
        if self.lead < 0 or self.horizon < 1 or self.train_stride < 1:
            raise ValueError("need lead >= 0, horizon >= 1, train_stride >= 1")

    def targets(self, origin: int) -> range:
        return range(origin + self.lead + 1, origin + self.lead + self.horizon + 1)

    def last_train_time(self, origin: int) -> int:
        return origin - self.lead - self.horizon

    def train_times(self, origin: int) -> np.ndarray:
        hi = self.last_train_time(origin)
        lo = self.min_history
        if self.window == "sliding":
            lo = max(lo, hi - self.window_length + 1)
        if hi < lo:
            return np.array([], dtype=np.int64)
        return np.arange(hi, lo - 1, -self.train_stride)[::-1]


def make_origins(
    first_target: int,
    last_target: int,
    lead: int,
    horizon: int,
    step: int,
    n_folds: int | None = None,
) -> tuple[int, ...]:
    """Origins advancing by ``step`` whose full target window lies in [first_target,
    last_target]; optionally only the first ``n_folds``."""
    first = first_target - lead - 1
    last = last_target - lead - horizon
    if last < first:
        raise ValueError("evaluation window shorter than one horizon")
    origins = tuple(range(first, last + 1, step))
    return origins[:n_folds] if n_folds else origins


def check_split(train: pl.DataFrame, test: pl.DataFrame, origin: int, spec: BacktestSpec) -> None:
    """Raise if a training example's target window or features extend past the origin, or if
    test rows are not exactly the origin's forecast window."""
    if train.height:
        if int(train["origin_idx"].max()) > spec.last_train_time(origin):  # type: ignore[arg-type]
            raise AssertionError(f"origin {origin}: training example with t > T - L - H")
        if int(train["target_idx"].max()) > origin:  # type: ignore[arg-type]
            raise AssertionError(f"origin {origin}: training target after the origin")
        n_steps = train.group_by(["series_idx", "origin_idx"]).len()["len"]
        if int(n_steps.min()) != spec.horizon:  # type: ignore[arg-type]
            raise AssertionError(f"origin {origin}: training example with a partial window")
    if test.height:
        if set(test["origin_idx"].unique().to_list()) != {origin}:
            raise AssertionError(f"origin {origin}: test rows from another origin")
        tg = test["target_idx"]
        lo, hi = origin + spec.lead + 1, origin + spec.lead + spec.horizon
        if int(tg.min()) < lo or int(tg.max()) > hi:  # type: ignore[arg-type]
            raise AssertionError(f"origin {origin}: test target outside (T+L, T+L+H]")
