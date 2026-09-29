"""CPU tests for DA-CIT p-values and the CRT formula."""

import jax
import jax.numpy as jnp

from da_diff.cit import crt_pvalue, da_cit
from da_diff.score import apply_score, init_score
from da_diff.train import train_da_diff


def test_pvalue_formula_canned():
    t_obs = 1.0
    t_b = jnp.array([0.5, 1.0, 1.5, 0.2], dtype=jnp.float32)
    p = float(crt_pvalue(t_obs, t_b))
    assert abs(p - 3.0 / 5.0) < 1e-6


def test_pvalue_at_least_one_over_one_plus_b():
    t_obs = 100.0
    t_b = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
    p = float(crt_pvalue(t_obs, t_b))
    assert abs(p - 1.0 / 4.0) < 1e-6


def _tiny_model(key, x, z, epochs=6):
    return train_da_diff(
        key,
        x,
        z,
        sources=[],
        hidden=32,
        epochs=epochs,
        warmup_epochs=epochs,
        lr=1e-2,
        time_dim=8,
    )


def test_cit_null_not_tiny():
    B, n = 20, 30
    ps = []
    for seed in (0, 1, 2):
        key = jax.random.PRNGKey(seed)
        key, kx, ky, kz, kt = jax.random.split(key, 5)
        x = jax.random.normal(kx, (n, 1), dtype=jnp.float32)
        y = jax.random.normal(ky, (n, 1), dtype=jnp.float32)
        z = jax.random.normal(kz, (n, 1), dtype=jnp.float32)
        out = _tiny_model(kt, x, z, epochs=4)
        cit = da_cit(out["params"], apply_score, key, x, y, z, B=B, n_steps=12)
        p = float(cit["p"])
        assert p >= 1.0 / (1.0 + B)
        ps.append(p)
    assert max(ps) > 0.05
    key = jax.random.PRNGKey(11)
    key, kx, ky, kz, kt = jax.random.split(key, 5)
    x = jax.random.normal(kx, (n, 1), dtype=jnp.float32)
    y = jax.random.normal(ky, (n, 1), dtype=jnp.float32)
    z = jax.random.normal(kz, (n, 1), dtype=jnp.float32)
    params = init_score(kt, 1, 1, hidden=32, time_dim=8)
    cit = da_cit(params, apply_score, key, x, y, z, B=B, n_steps=8)
    p_ind = float(cit["p"])
    assert p_ind >= 1.0 / (1.0 + B)
    assert p_ind > 0.1


def test_cit_alternative_smaller_than_null():
    B, n = 20, 30
    key = jax.random.PRNGKey(7)
    key, kx, kz, knull, kalt, kt = jax.random.split(key, 6)
    x = jax.random.normal(kx, (n, 1), dtype=jnp.float32)
    z = jax.random.normal(kz, (n, 1), dtype=jnp.float32)
    y_null = jax.random.normal(knull, (n, 1), dtype=jnp.float32)
    y_alt = x
    out = _tiny_model(kt, x, z, epochs=5)
    p_null = float(da_cit(out["params"], apply_score, kalt, x, y_null, z, B=B, n_steps=12)["p"])
    key, k2 = jax.random.split(key)
    p_alt = float(da_cit(out["params"], apply_score, k2, x, y_alt, z, B=B, n_steps=12)["p"])
    assert p_null >= 1.0 / (1.0 + B)
    assert p_alt >= 1.0 / (1.0 + B)
    assert p_alt < p_null or p_alt <= 0.2
