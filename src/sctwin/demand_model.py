"""Step 5: Bayesian latent-demand model ladder (NumPyro).

Hourly demand in the day window (hours 6-21) of series s, day d, hour h:

    eta[s,d,h] = alpha + a[s] + gamma[h] + g[s,h] + dow[d] + b_disc x_disc[s,d]
                 + b_act x_act[s,d] + u[s,d]
    P(D > 0) = sigmoid(k0 + k1 (eta - alpha))
    log D | D > 0 ~ Normal(m, sigma_y^2),  m chosen so that E[D] = exp(eta)

Sales are normalized, continuous values (docs/data_card.md), so the likelihood is a continuous
hurdle-lognormal rather than a count model. (A hurdle-Gamma was tried first; the gradient of
its survival function with respect to the shape made NUTS about 10x slower.) All random
effects are non-centred.

Observation model per hourly availability class (``sctwin.panel.availability_classes``):

    ==========  ==================  =====================================================
    treatment   meaning             log-likelihood contribution
    ==========  ==================  =====================================================
    exact       D = y               log f(y)
    censored    D >= y (lower bnd)  log P(D >= y)   (0 when y = 0)
    drop        no information      0
    ==========  ==================  =====================================================

Ladder:
    M0  static hierarchical, i.i.d. day effects; stockout hours treated as missing
        (A, P_pre exact; P_first, U dropped)
    M1  M0 + censoring: P_pre and P_first right-censored at the observed sales, U dropped
    M2  M1 + AR(1) day effects u[s,d] = rho u[s,d-1] + sigma_u e[s,d]
    M3  M2 on more series (same code, larger data)

Reconstruction of a censored day (Step 4 benchmark) is made as of the end of that day: global
parameters come from a posterior fitted on days strictly before it; the day's own effect
u[s,d] is inferred from that day's censored observations alone (grid posterior per draw); the
demand of censored hours is drawn from the predictive distribution truncated at the observation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import log_ndtr
from scipy import special as sps

from sctwin.panel import CLS_A, CLS_P_FIRST, CLS_P_PRE, CLS_U, DAY_HOURS, availability_classes

jax.config.update("jax_enable_x64", True)

LO, HI = DAY_HOURS
NH = HI - LO
EXACT, CENSORED, DROP = 0, 1, 2
TREATMENT_CODES = {"exact": EXACT, "censored": CENSORED, "drop": DROP}
CLASS_NAMES = {CLS_A: "A", CLS_P_PRE: "P_pre", CLS_P_FIRST: "P_first", CLS_U: "U"}
DERIVED = ("a", "gamma", "g", "dow_eff", "u")
SCALARS = ("alpha", "sigma_a", "tau_g", "sigma_u", "b_disc", "b_act", "sigma_y", "k0", "k1")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    treatment: dict[str, str]  # class name -> exact | censored | drop
    dynamic: bool = False

    def codes(self) -> np.ndarray:
        """Treatment code per class code (indexed by CLS_*)."""
        out = np.full(5, DROP, dtype=np.int8)
        for c, n in CLASS_NAMES.items():
            out[c] = TREATMENT_CODES[self.treatment[n]]
        return out


LADDER = {
    "M0": ModelSpec("M0", {"A": "exact", "P_pre": "exact", "P_first": "drop", "U": "drop"}),
    "M1": ModelSpec("M1", {"A": "exact", "P_pre": "censored", "P_first": "censored", "U": "drop"}),
    "M2": ModelSpec(
        "M2", {"A": "exact", "P_pre": "censored", "P_first": "censored", "U": "drop"}, True
    ),
}


# --------------------------------------------------------------------------- data


@dataclass
class ModelData:
    y: np.ndarray  # (S, D, NH) observed sales in hours 6-21
    cls: np.ndarray  # (S, D, NH) availability class codes
    dow: np.ndarray  # (D,) weekday 0-6
    x_disc: np.ndarray  # (S, D) 1 - discount (0 where the discount is invalid)
    x_act: np.ndarray  # (S, D) activity flag
    series: np.ndarray  # (S,) panel series index
    days: np.ndarray  # (D,) panel day index (consecutive)

    @property
    def shape(self) -> tuple[int, int]:
        return self.y.shape[0], self.y.shape[1]


def discount_feature(discount: np.ndarray) -> np.ndarray:
    ok = np.isfinite(discount) & (discount > 0) & (discount <= 1.5)
    return np.where(ok, 1.0 - discount, 0.0)


def make_model_data(
    sales: np.ndarray,
    stock: np.ndarray,
    discount: np.ndarray,
    activity: np.ndarray,
    weekday: np.ndarray,
    series: np.ndarray,
    days: np.ndarray,
) -> ModelData:
    """Model inputs for consecutive ``days`` of ``series`` (panel arrays are (S, D, 24))."""
    days = np.asarray(days)
    if days.size == 0 or not (np.diff(days) == 1).all():
        raise ValueError("days must be consecutive")
    ix = np.ix_(series, days)
    return ModelData(
        y=sales[ix][..., LO:HI].astype(np.float64),
        cls=availability_classes(stock[ix])[..., LO:HI],
        dow=np.asarray(weekday)[days].astype(np.int32),
        x_disc=discount_feature(discount[ix]),
        x_act=activity[ix].astype(np.float64),
        series=np.asarray(series),
        days=days,
    )


# --------------------------------------------------------------------------- likelihood


def hurdle_terms(eta: Any, alpha: Any, sigma_y: Any, k0: Any, k1: Any, xp: Any = jnp) -> tuple:
    """(log P(D>0), log P(D=0), m) where log D | D > 0 ~ N(m, sigma_y^2) and E[D] = exp(eta)."""
    z = k0 + k1 * (eta - alpha)
    log_p = -xp.logaddexp(0.0, -z)
    log_q = -xp.logaddexp(0.0, z)
    m = eta - log_p - 0.5 * sigma_y**2
    return log_p, log_q, m


def loglik(y: Any, code: Any, eta: Any, alpha: Any, sigma_y: Any, k0: Any, k1: Any) -> Any:
    """Pointwise log-likelihood (JAX) under the per-observation treatment codes."""
    log_p, log_q, m = hurdle_terms(eta, alpha, sigma_y, k0, k1)
    pos = y > 0
    ly = jnp.log(jnp.where(pos, y, 1.0))
    r = (ly - m) / sigma_y
    log_dens = -0.5 * r**2 - jnp.log(sigma_y) - 0.5 * jnp.log(2 * jnp.pi) - ly
    exact = jnp.where(pos, log_p + log_dens, log_q)
    cens = jnp.where(pos, log_p + log_ndtr(-r), 0.0)
    return jnp.where(code == EXACT, exact, jnp.where(code == CENSORED, cens, 0.0))


_loglik_jit = jax.jit(loglik)


# --------------------------------------------------------------------------- model


def _centred(x: Any, axis: int = -1) -> Any:
    return x - x.mean(axis=axis, keepdims=True)


def ar1(raw: Any, rho: Any, sigma: Any) -> Any:
    """AR(1) paths over the last axis from standard normal innovations (stationary start)."""
    first = raw[:, 0] * sigma / jnp.sqrt(1 - rho**2)

    def step(prev: Any, e: Any) -> tuple[Any, Any]:
        cur = rho * prev + sigma * e
        return cur, cur

    _, rest = jax.lax.scan(step, first, raw[:, 1:].T)
    return jnp.concatenate([first[:, None], rest.T], axis=1)


def linear_predictor(p: dict[str, Any], dow: Any, x_disc: Any, x_act: Any) -> Any:
    """eta without the day effect: (S, D, NH)."""
    return (
        p["alpha"]
        + p["a"][:, None, None]
        + p["gamma"][None, None, :]
        + p["g"][:, None, :]
        + p["dow_eff"][dow][None, :, None]
        + (p["b_disc"] * x_disc + p["b_act"] * x_act)[:, :, None]
    )


def demand_model(
    y: Any, code: Any, dow: Any, x_disc: Any, x_act: Any, dynamic: bool = False
) -> None:
    n_s, n_d = x_disc.shape
    n_h = NH
    alpha = numpyro.sample("alpha", dist.Normal(-2.0, 1.5))
    sigma_a = numpyro.sample("sigma_a", dist.HalfNormal(1.0))
    a = numpyro.deterministic(
        "a", sigma_a * numpyro.sample("a_raw", dist.Normal(0, 1).expand([n_s]))
    )
    gamma = numpyro.deterministic(
        "gamma", _centred(numpyro.sample("gamma_raw", dist.Normal(0, 1).expand([n_h])))
    )
    tau_g = numpyro.sample("tau_g", dist.HalfNormal(0.5))
    g_raw = numpyro.sample("g_raw", dist.Normal(0, 1).expand([n_s, n_h]))
    g = numpyro.deterministic("g", tau_g * _centred(_centred(g_raw, -1), 0))
    dow_eff = numpyro.deterministic(
        "dow_eff", _centred(numpyro.sample("dow_raw", dist.Normal(0, 0.5).expand([7])))
    )
    b_disc = numpyro.sample("b_disc", dist.Normal(0, 1.0))
    b_act = numpyro.sample("b_act", dist.Normal(0, 0.5))
    sigma_u = numpyro.sample("sigma_u", dist.HalfNormal(0.5))
    u_raw = numpyro.sample("u_raw", dist.Normal(0, 1).expand([n_s, n_d]))
    if dynamic:
        rho = numpyro.sample("rho", dist.Beta(2.0, 2.0))
        u = numpyro.deterministic("u", ar1(u_raw, rho, sigma_u))
    else:
        u = numpyro.deterministic("u", sigma_u * u_raw)
    sigma_y = numpyro.sample("sigma_y", dist.HalfNormal(1.5))
    k0 = numpyro.sample("k0", dist.Normal(0.0, 2.0))
    k1 = numpyro.sample("k1", dist.Normal(1.0, 1.0))
    p = {"alpha": alpha, "a": a, "gamma": gamma, "g": g, "dow_eff": dow_eff,
         "b_disc": b_disc, "b_act": b_act}  # fmt: skip
    eta = linear_predictor(p, dow, x_disc, x_act) + u[:, :, None]
    if y is not None:
        numpyro.factor("obs", loglik(y, code, eta, alpha, sigma_y, k0, k1).sum())


def model_args(data: ModelData, spec: ModelSpec) -> dict[str, Any]:
    return {
        "y": jnp.asarray(data.y),
        "code": jnp.asarray(spec.codes()[data.cls]),
        "dow": jnp.asarray(data.dow),
        "x_disc": jnp.asarray(data.x_disc),
        "x_act": jnp.asarray(data.x_act),
        "dynamic": spec.dynamic,
    }


# --------------------------------------------------------------------------- fitting


@dataclass
class FitResult:
    spec: ModelSpec
    samples: dict[str, np.ndarray]  # flattened over chains: (draws, ...)
    diagnostics: dict[str, Any]
    by_chain: dict[str, np.ndarray] = field(repr=False, default_factory=dict)


def fit(
    data: ModelData,
    spec: ModelSpec,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    target_accept: float = 0.9,
    max_tree_depth: int = 10,
    chain_method: str = "parallel",
    progress: bool = False,
) -> FitResult:
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS

    kernel = NUTS(demand_model, target_accept_prob=target_accept, max_tree_depth=max_tree_depth)
    if chain_method == "parallel" and jax.local_device_count() < num_chains:
        chain_method = "sequential"
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                chain_method=chain_method, progress_bar=progress)  # fmt: skip
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed), **model_args(data, spec), extra_fields=("diverging",))
    runtime = time.perf_counter() - t0
    grouped = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    # convergence is judged on the sampled (non-centred) coordinates, not on derived quantities
    summ = summary({k: v for k, v in grouped.items() if k not in DERIVED}, group_by_chain=True)
    rhat = {k: float(np.nanmax(v["r_hat"])) for k, v in summ.items()}
    ess = {k: float(np.nanmin(v["n_eff"])) for k, v in summ.items()}
    div = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    diag = {
        "num_warmup": num_warmup, "num_samples": num_samples, "num_chains": num_chains,
        "runtime_s": round(runtime, 1), "divergences": div,
        "max_rhat": max(rhat.values()), "min_ess": min(ess.values()),
        "rhat": rhat, "ess": ess,
        "scalars": {k: {"mean": float(summ[k]["mean"]), "sd": float(summ[k]["std"]),
                        "rhat": rhat[k], "ess": ess[k]}
                    for k in (*SCALARS, "rho") if k in summ},
    }  # fmt: skip
    flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in grouped.items()}
    return FitResult(spec, flat, diag, grouped)


# --------------------------------------------------------------------------- checks


def pointwise_loglik(
    res: FitResult, data: ModelData, classes: tuple[int, ...] = (CLS_A,), max_draws: int = 1000
) -> np.ndarray:
    """(draws, n_obs) log-likelihood of the exact-treated observations in ``classes``."""
    s = res.samples
    idx = np.linspace(0, len(s["alpha"]) - 1, min(max_draws, len(s["alpha"]))).astype(int)
    mask = np.isin(data.cls, classes)
    code = jnp.zeros(int(mask.sum()), dtype=jnp.int8)
    y = jnp.asarray(data.y[mask])

    def one(i: int) -> Any:
        p = {k: s[k][i] for k in ("alpha", "a", "gamma", "g", "dow_eff", "b_disc", "b_act")}
        eta = linear_predictor(p, data.dow, data.x_disc, data.x_act) + s["u"][i][:, :, None]
        return loglik(y, code, eta[mask], s["alpha"][i], s["sigma_y"][i], s["k0"][i], s["k1"][i])

    return np.stack([np.asarray(one(int(i))) for i in idx])


def sample_hurdle(
    eta: np.ndarray, alpha: Any, sigma_y: Any, k0: Any, k1: Any, rng: np.random.Generator
) -> np.ndarray:
    log_p, _, m = hurdle_terms(eta, alpha, sigma_y, k0, k1, xp=np)
    pos = rng.random(eta.shape) < np.exp(log_p)
    return np.where(pos, np.exp(m + sigma_y * rng.standard_normal(eta.shape)), 0.0)


def sample_truncated(
    eta: np.ndarray, lower: np.ndarray, alpha: Any, sigma_y: Any, k0: Any, k1: Any,
    rng: np.random.Generator,
) -> np.ndarray:  # fmt: skip
    """Draws of D | D >= lower (lower > 0): the positive lognormal part truncated below,
    by inverting the survival function in log space (stable far in the tail)."""
    _, _, m = hurdle_terms(eta, alpha, sigma_y, k0, k1, xp=np)
    r0 = (np.log(lower) - m) / sigma_y
    log_q = sps.log_ndtr(-r0)  # log P(Z >= r0)
    log_w = log_q + np.log(rng.random(eta.shape))  # uniform on (0, q) in log space
    r = -sps.ndtri_exp(log_w)
    return np.maximum(np.exp(m + sigma_y * r), lower)


def predictive_checks(
    res: FitResult, data: ModelData, n_draws: int = 200, seed: int = 0
) -> dict[str, Any]:
    """Posterior predictive check on fully available (A) hours: statistics of the data vs the
    replicated data, with the share of replicates above the observed value (PPC p-value)."""
    rng = np.random.default_rng(seed)
    s = res.samples
    idx = rng.choice(len(s["alpha"]), size=min(n_draws, len(s["alpha"])), replace=False)
    mask = data.cls == CLS_A
    y = data.y[mask]

    def stats(v: np.ndarray) -> dict[str, float]:
        return {"zero_frac": float((v == 0).mean()), "mean": float(v.mean()),
                "q50": float(np.quantile(v, 0.5)), "q90": float(np.quantile(v, 0.9)),
                "q99": float(np.quantile(v, 0.99)), "max": float(v.max())}  # fmt: skip

    reps = []
    for i in idx:
        p = {k: s[k][i] for k in ("alpha", "a", "gamma", "g", "dow_eff", "b_disc", "b_act")}
        eta = np.asarray(linear_predictor(p, data.dow, data.x_disc, data.x_act))
        eta = eta + s["u"][i][:, :, None]
        reps.append(stats(sample_hurdle(eta[mask], s["alpha"][i], s["sigma_y"][i], s["k0"][i],
                                        s["k1"][i], rng)))  # fmt: skip
    obs = stats(y)
    out = {}
    for k, v in obs.items():
        r = np.array([x[k] for x in reps])
        lo, hi = np.quantile(r, [0.05, 0.95])
        out[k] = {"observed": v, "rep_q05": float(lo), "rep_q95": float(hi),
                  "ppc_p": float((r >= v).mean())}  # fmt: skip
    return out


# --------------------------------------------------------------------------- reconstruction


@dataclass
class DayRows:
    """Censored days to reconstruct: one row per (series, day), hours 6-21."""

    s_pos: np.ndarray  # (R,) position of the series in the fitted ModelData
    day: np.ndarray  # (R,) panel day index (must be after every fitted day)
    y: np.ndarray  # (R, NH) observed sales
    cls: np.ndarray  # (R, NH) availability classes from the observed flags
    dow: np.ndarray  # (R,)
    x_disc: np.ndarray  # (R,)
    x_act: np.ndarray  # (R,)


def reconstruct(
    res: FitResult,
    data: ModelData,
    rows: DayRows,
    n_draws: int = 400,
    grid: int = 121,
    seed: int = 0,
    chunk: int = 64,
) -> np.ndarray:
    """Posterior draws (R, n_draws, NH) of demand on each row's hours 6-21, as of the end of
    that day. Raises if a row's day is not strictly after the fitted window (point-in-time)."""
    if (rows.day <= data.days.max()).any():
        raise ValueError("reconstructed days must be after every day used for fitting")
    rng = np.random.default_rng(seed)
    s = res.samples
    idx = np.linspace(0, len(s["alpha"]) - 1, n_draws).astype(int)
    codes = res.spec.codes()[rows.cls]  # (R, NH)
    z = np.linspace(-6.0, 6.0, grid)
    out = np.empty((rows.y.shape[0], n_draws, NH))
    for j, i in enumerate(idx):
        p = {k: s[k][i] for k in ("alpha", "a", "gamma", "g", "dow_eff", "b_disc", "b_act")}
        al, sh, k0, k1 = (float(s[k][i]) for k in ("alpha", "sigma_y", "k0", "k1"))
        base = (p["alpha"] + p["a"][rows.s_pos][:, None] + p["gamma"][None, :]
                + p["g"][rows.s_pos] + p["dow_eff"][rows.dow][:, None]
                + (p["b_disc"] * rows.x_disc + p["b_act"] * rows.x_act)[:, None])  # fmt: skip
        su = float(s["sigma_u"][i])
        if res.spec.dynamic:
            rho = float(s["rho"][i])
            k = rows.day - data.days.max()
            m = rho**k * s["u"][i][rows.s_pos, -1]
            sd = su * np.sqrt((1 - rho ** (2 * k)) / (1 - rho**2))
        else:
            m, sd = np.zeros(len(rows.day)), np.full(len(rows.day), su)
        for c0 in range(0, len(rows.day), chunk):
            sl = slice(c0, c0 + chunk)
            u = m[sl, None] + sd[sl, None] * z[None, :]  # (r, G)
            eta = base[sl, None, :] + u[:, :, None]  # (r, G, NH)
            yy, cc = jnp.asarray(rows.y[sl, None, :]), jnp.asarray(codes[sl, None, :])
            ll = np.asarray(_loglik_jit(yy, cc, jnp.asarray(eta), al, sh, k0, k1)).sum(-1)
            lw = ll - 0.5 * z[None, :] ** 2
            w = np.exp(lw - lw.max(axis=1, keepdims=True))
            cdf = np.cumsum(w, axis=1)
            cdf /= cdf[:, -1:]
            pick = (cdf < rng.random((cdf.shape[0], 1))).sum(axis=1)
            zz = z[np.minimum(pick, grid - 1)] + (rng.random(pick.shape) - 0.5) * (z[1] - z[0])
            e = base[sl] + (m[sl] + sd[sl] * zz)[:, None]  # (r, NH)
            yv, cv = rows.y[sl], codes[sl]
            draw = sample_hurdle(e, al, sh, k0, k1, rng)
            trunc = (cv == CENSORED) & (yv > 0)
            if trunc.any():
                draw[trunc] = sample_truncated(e[trunc], yv[trunc], al, sh, k0, k1, rng)
            draw = np.where(cv == EXACT, yv, np.maximum(draw, yv))
            out[sl, j] = draw
    return out


# --------------------------------------------------------------------------- simulation


def simulate(
    params: dict[str, Any], dow: np.ndarray, x_disc: np.ndarray, x_act: np.ndarray, seed: int
) -> np.ndarray:
    """Demand (S, D, NH) from the model with given parameters (a, gamma, g, dow_eff, u, ...)."""
    eta = np.asarray(linear_predictor(params, dow, x_disc, x_act)) + params["u"][:, :, None]
    rng = np.random.default_rng(seed)
    return sample_hurdle(eta, params["alpha"], params["sigma_y"], params["k0"], params["k1"], rng)


def prior_draw(
    n_series: int, n_days: int, dynamic: bool, seed: int, x_disc: np.ndarray, x_act: np.ndarray,
    dow: np.ndarray,
) -> dict[str, Any]:  # fmt: skip
    """One draw of all parameters from the prior (for SBC and prior predictive checks)."""
    from numpyro.infer import Predictive

    pred = Predictive(demand_model, num_samples=1)
    d = pred(jax.random.PRNGKey(seed), y=None, code=None, dow=jnp.asarray(dow),
             x_disc=jnp.asarray(x_disc), x_act=jnp.asarray(x_act), dynamic=dynamic)  # fmt: skip
    return {k: np.asarray(v)[0] for k, v in d.items()}
