"""CPU tests for DA-Diff: DSM, weighted ERM, reverse shape, forward prior."""

import jax
import jax.numpy as jnp

from da_diff.score import apply_score, init_score
from da_diff.sde import forward_noise, reverse_sample
from da_diff.train import (
    _adam_init,
    _adam_update,
    dsm_loss,
    source_weights_from_dsm,
    train_da_diff,
    weighted_dsm_loss,
)


def test_dsm_loss_decreases():
    key = jax.random.PRNGKey(0)
    n, dx, dz = 128, 2, 2
    key, kz, kn, ki = jax.random.split(key, 4)
    z = jax.random.normal(kz, (n, dz), dtype=jnp.float32)
    x = z + 0.25 * jax.random.normal(kn, (n, dx), dtype=jnp.float32)
    params = init_score(ki, dx, dz, hidden=32, time_dim=8)
    opt = _adam_init(params, 1e-2)

    def step(params, opt, k):
        loss, grads = jax.value_and_grad(lambda p: dsm_loss(p, k, x, z))(params)
        params, opt = _adam_update(params, grads, opt)
        return params, opt, loss

    key, k0 = jax.random.split(key)
    loss0 = float(dsm_loss(params, k0, x, z))
    for _ in range(40):
        key, k_step = jax.random.split(key)
        params, opt, loss = step(params, opt, k_step)
    assert jnp.isfinite(loss)
    assert float(loss) < loss0


def test_weighted_erm_warmup_and_weights():
    key = jax.random.PRNGKey(1)
    n0, n1, n2 = 32, 40, 24
    dx = dz = 1
    key, k0, k1, k2 = jax.random.split(key, 4)

    def _pair(k, n, shift):
        kz, kn = jax.random.split(k)
        z = shift + jax.random.normal(kz, (n, dz), dtype=jnp.float32)
        x = 0.6 * z + 0.3 * jax.random.normal(kn, (n, dx), dtype=jnp.float32)
        return x, z

    x0, z0 = _pair(k0, n0, 0.0)
    sources = [_pair(k1, n1, 0.8), _pair(k2, n2, -0.5)]
    out = train_da_diff(
        key,
        x0,
        z0,
        sources,
        hidden=32,
        epochs=8,
        warmup_epochs=3,
        lr=5e-3,
        update_every=2,
        time_dim=8,
    )
    w = jnp.asarray(out["weights"])
    assert w.shape == (2,)
    assert jnp.all(w >= 0)
    assert jnp.isfinite(jnp.asarray(out["history"][-1]))
    ns = jnp.array([n1, n2], dtype=jnp.float32)
    W_N = float(jnp.sum(w * ns))
    assert abs(float(out["W_N"]) - W_N) < 1e-4

    key, kl = jax.random.split(key)
    loss, aux = weighted_dsm_loss(out["params"], kl, x0, z0, sources, w)
    assert jnp.isfinite(loss)
    assert jnp.all(aux["weights"] >= 0)
    assert abs(float(aux["W_N"]) - float(jnp.sum(w * ns))) < 1e-4


def test_reverse_sample_shape():
    key = jax.random.PRNGKey(2)
    n, dx, dz, B = 16, 2, 3, 5
    key, kz, ki = jax.random.split(key, 3)
    z = jax.random.normal(kz, (n, dz), dtype=jnp.float32)
    params = init_score(ki, dx, dz, hidden=32, time_dim=8)
    xs = reverse_sample(params, apply_score, key, z, dx, n_steps=10, n_samples=B)
    assert xs.shape == (B, n, dx)
    assert jnp.isfinite(xs).all()


def test_forward_noising_t1_near_prior():
    key = jax.random.PRNGKey(3)
    x0 = jnp.ones((2048, 2), dtype=jnp.float32) * 3.0
    xt, _ = forward_noise(key, x0, 1.0)
    assert xt.shape == x0.shape
    assert abs(float(jnp.mean(xt))) < 0.25


def test_source_weights_nonnegative():
    w = source_weights_from_dsm(1.0, jnp.array([1.0, 3.0]), jnp.array([10.0, 20.0]))
    assert jnp.all(w >= 0)
    assert abs(float(jnp.mean(w)) - 1.0) < 1e-5
    assert w[0] > w[1]
