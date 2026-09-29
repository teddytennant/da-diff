"""Tiny CPU demo: DA-Diff on two source domains, then one DA-CIT p-value."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from da_diff._host import preload

preload()

import jax
import jax.numpy as jnp

from da_diff.cit import da_cit
from da_diff.score import apply_score
from da_diff.train import train_da_diff


def make_domain(key, n, shift, scale=1.0, alt=False):
    kz, knx, kny = jax.random.split(key, 3)
    z = shift + scale * jax.random.normal(kz, (n, 1), dtype=jnp.float32)
    x = jnp.tanh(z) + 0.3 * jax.random.normal(knx, (n, 1), dtype=jnp.float32)
    if alt:
        y = x + 0.2 * jax.random.normal(kny, (n, 1), dtype=jnp.float32)
    else:
        y = jnp.tanh(z) + 0.3 * jax.random.normal(kny, (n, 1), dtype=jnp.float32)
    return x, y, z


def main():
    key = jax.random.PRNGKey(0)
    key, k0, k1, k2, kt, kc = jax.random.split(key, 6)
    x0, y0, z0 = make_domain(k0, 64, 0.0, alt=False)
    x1, _, z1 = make_domain(k1, 64, 0.7, scale=1.2, alt=False)
    x2, _, z2 = make_domain(k2, 48, -0.4, scale=0.8, alt=False)
    out = train_da_diff(
        kt,
        x0,
        z0,
        sources=[(x1, z1), (x2, z2)],
        hidden=32,
        epochs=12,
        warmup_epochs=4,
        lr=5e-3,
        update_every=4,
        time_dim=8,
    )
    print("warmup+weighted history (last 5):", [round(h, 4) for h in out["history"][-5:]])
    print("source weights:", [float(w) for w in out["weights"]], "W_N=", float(out["W_N"]))
    cit = da_cit(out["params"], apply_score, kc, x0, y0, z0, B=16, n_steps=16)
    print("DA-CIT p-value (null-ish y|z):", float(cit["p"]))

    key, ka, kca = jax.random.split(key, 3)
    xa, ya, za = make_domain(ka, 64, 0.0, alt=True)
    cit_a = da_cit(out["params"], apply_score, kca, xa, ya, za, B=16, n_steps=16)
    print("DA-CIT p-value (alt y~x):", float(cit_a["p"]))


if __name__ == "__main__":
    main()
