"""DA-CIT: CRT p-value with a reverse-sampled conditional (Alg. 2, eq. 3).

The paper uses distance correlation (Székely 2007) as T(X, Y, Z). This kernel
uses biased distance covariance of linearly residualized (X, Y) | Z (partial
dCov): OLS-regress X and Y on (1, Z), then dCov^2 of the residuals. An
unconditional dCov(X, Y) that ignores Z is also provided. Split of D0 into
estimation / testing halves is not required by the API.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from da_diff.sde import GAMMA, T0, reverse_sample


def _as_2d(a):
    a = jnp.asarray(a, dtype=jnp.float32)
    if a.ndim == 1:
        a = a[:, None]
    return a


def _pairwise_dist(a):
    diff = a[:, None, :] - a[None, :, :]
    return jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-12)


def _double_center(d):
    row = jnp.mean(d, axis=1, keepdims=True)
    col = jnp.mean(d, axis=0, keepdims=True)
    return d - row - col + jnp.mean(d)


def distance_covariance(x, y):
    """Biased dCov^2(X, Y) (Székely 2007), ignoring Z."""
    x, y = _as_2d(x), _as_2d(y)
    A = _double_center(_pairwise_dist(x))
    B = _double_center(_pairwise_dist(y))
    return jnp.mean(A * B)


def _ols_resid(a, z):
    a, z = _as_2d(a), _as_2d(z)
    n = a.shape[0]
    Z = jnp.concatenate([jnp.ones((n, 1), dtype=a.dtype), z], axis=1)
    beta, *_ = jnp.linalg.lstsq(Z, a, rcond=None)
    return a - Z @ beta


def partial_distance_covariance(x, y, z):
    """dCov^2 of OLS residuals of X, Y on Z (partial dCov)."""
    return distance_covariance(_ols_resid(x, z), _ols_resid(y, z))


def crt_pvalue(t_obs, t_null):
    """p = (1 + sum 1{T^{(b)} >= T}) / (1 + B)  (eq. 3)."""
    t_null = jnp.asarray(t_null, dtype=jnp.float32)
    t_obs = jnp.asarray(t_obs, dtype=jnp.float32)
    b = t_null.shape[0]
    count = jnp.sum(t_null >= t_obs)
    return (1.0 + count) / (1.0 + b)


def dependence_stat(x, y, z, statistic: str = "partial_dcov"):
    if statistic == "dcov":
        return distance_covariance(x, y)
    if statistic == "partial_dcov":
        return partial_distance_covariance(x, y, z)
    raise ValueError(f"unknown statistic {statistic}")


def da_cit(
    params,
    apply_fn,
    key,
    x,
    y,
    z,
    B: int = 20,
    n_steps: int = 20,
    statistic: str = "partial_dcov",
    t0: float = T0,
    gamma: float = GAMMA,
    method: str = "ddpm",
):
    """Algorithm 2: resample X | Z, compare T to the CRT null.

    Returns dict with p, t_obs, t_null, x_null.
    """
    x = _as_2d(x)
    y = _as_2d(y)
    z = _as_2d(z)
    dx = x.shape[-1]
    t_obs = dependence_stat(x, y, z, statistic=statistic)
    x_null = reverse_sample(
        params,
        apply_fn,
        key,
        z,
        dx,
        n_steps=n_steps,
        n_samples=B,
        t0=t0,
        gamma=gamma,
        method=method,
    )

    def _t_b(xb):
        return dependence_stat(xb, y, z, statistic=statistic)

    t_null = jax.vmap(_t_b)(x_null)
    p = crt_pvalue(t_obs, t_null)
    return {"p": p, "t_obs": t_obs, "t_null": t_null, "x_null": x_null}
