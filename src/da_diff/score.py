"""Tiny ReLU MLP score network s_theta(x, z, t).

Paper eq. 6 uses a ReLU class F. This kernel uses a 2-hidden-layer MLP of
width 32--64 with frozen Gaussian Fourier time features, not the paper's
architecture.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def init_score(key, dx: int, dz: int, hidden: int = 64, time_dim: int = 16, scale: float = 0.2):
    k1, k2, k3, kf = jax.random.split(key, 4)
    in_dim = dx + dz + 2 * time_dim
    params = {
        "W1": scale * jax.random.normal(k1, (in_dim, hidden), dtype=jnp.float32),
        "b1": jnp.zeros((hidden,), dtype=jnp.float32),
        "W2": scale * jax.random.normal(k2, (hidden, hidden), dtype=jnp.float32),
        "b2": jnp.zeros((hidden,), dtype=jnp.float32),
        "W3": scale * jax.random.normal(k3, (hidden, dx), dtype=jnp.float32),
        "b3": jnp.zeros((dx,), dtype=jnp.float32),
        "freqs": jax.random.normal(kf, (time_dim,), dtype=jnp.float32) * 16.0,
    }
    return params


def _time_embed(t, freqs):
    t = jnp.asarray(t, dtype=jnp.float32)
    ang = t[..., None] * freqs
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)


def apply_score(params, x, z, t):
    """s_theta(x, z, t) with x (..., dx), z (..., dz), t scalar or (...,)."""
    x = jnp.asarray(x, dtype=jnp.float32)
    z = jnp.asarray(z, dtype=jnp.float32)
    t = jnp.asarray(t, dtype=jnp.float32)
    batch_shape = x.shape[:-1]
    t = jnp.broadcast_to(t, batch_shape)
    z = jnp.broadcast_to(z, batch_shape + z.shape[-1:])
    te = _time_embed(t, params["freqs"])
    h = jnp.concatenate([x, z, te], axis=-1)
    h = jax.nn.relu(h @ params["W1"] + params["b1"])
    h = jax.nn.relu(h @ params["W2"] + params["b2"])
    return h @ params["W3"] + params["b3"]
