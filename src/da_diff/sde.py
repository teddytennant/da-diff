"""Forward VP process, DSM target, and reverse sampler (paper eqs. 4--6).

The paper uses the OU / VP SDE on t in [0, T]
    dX = -(1/2) X dt + dB,
    a_t = exp(-t/2),  sigma_t = sqrt(1 - exp(-t)).
Here time is mapped to the unit interval (0, 1] with a log-linear VP schedule
    alpha(t) = exp(-gamma t),  sigma(t) = sqrt(1 - alpha(t)^2),
which is the same family with time scaled by 2*gamma. Default gamma=4 so that
t=1 is near the N(0, I) prior (paper: T large). The DSM target is
    nabla_{x_t} log p(x_t | x_0) = -(x_t - alpha x_0) / sigma^2 = -eps / sigma
(eq. 6). Reverse sampling is DDPM-style ancestral (eq. 5 analogue) with an
Euler--Maruyama option.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

GAMMA = 4.0
T0 = 1e-3


def alpha_sigma(t, gamma: float = GAMMA):
    """VP coefficients alpha(t), sigma(t) for t in (0, 1]."""
    t = jnp.asarray(t, dtype=jnp.float32)
    t = jnp.clip(t, 0.0, 1.0)
    alpha = jnp.exp(-gamma * t)
    sigma = jnp.sqrt(jnp.clip(1.0 - alpha * alpha, 1e-12, 1.0))
    return alpha, sigma


def dsm_target(eps, sigma):
    """Score of p(x_t | x_0): -eps / sigma (eq. 6)."""
    sigma = jnp.clip(jnp.asarray(sigma, dtype=jnp.float32), 1e-4, None)
    return -eps / sigma


def forward_noise(key, x0, t, gamma: float = GAMMA):
    """x_t = alpha(t) x_0 + sigma(t) eps, eps ~ N(0, I). Returns (x_t, eps)."""
    x0 = jnp.asarray(x0, dtype=jnp.float32)
    t = jnp.asarray(t, dtype=jnp.float32)
    eps = jax.random.normal(key, x0.shape, dtype=jnp.float32)
    alpha, sigma = alpha_sigma(t, gamma=gamma)
    alpha, sigma = _match_batch(alpha, x0), _match_batch(sigma, x0)
    return alpha * x0 + sigma * eps, eps


def _match_batch(coeff, x):
    coeff = jnp.asarray(coeff, dtype=x.dtype)
    while coeff.ndim < x.ndim:
        coeff = coeff[..., None]
    return coeff


def reverse_sample(
    params,
    apply_fn,
    key,
    z,
    dx: int,
    n_steps: int = 25,
    n_samples: int = 1,
    t0: float = T0,
    gamma: float = GAMMA,
    method: str = "ddpm",
):
    """Draw X | Z from the learned reverse process.

    z: (n, dz). Returns (n_samples, n, dx).
    method: "ddpm" (ancestral) or "em" (Euler--Maruyama on the reverse SDE).
    """
    z = jnp.asarray(z, dtype=jnp.float32)
    n, dz = z.shape
    key, k0 = jax.random.split(key)
    x = jax.random.normal(k0, (n_samples, n, dx), dtype=jnp.float32)
    z_b = jnp.broadcast_to(z[None, :, :], (n_samples, n, dz))
    ts = jnp.linspace(1.0, t0, n_steps + 1, dtype=jnp.float32)

    def ddpm_body(i, carry):
        x, key = carry
        key, kn = jax.random.split(key)
        t = ts[i]
        t_next = ts[i + 1]
        alpha, sigma = alpha_sigma(t, gamma=gamma)
        alpha_n, sigma_n = alpha_sigma(t_next, gamma=gamma)
        score = apply_fn(params, x, z_b, t)
        x0_hat = (x + (sigma * sigma) * score) / jnp.clip(alpha, 1e-3)
        x0_hat = jnp.clip(x0_hat, -8.0, 8.0)
        eps = jax.random.normal(kn, x.shape, dtype=jnp.float32)
        x = alpha_n * x0_hat + sigma_n * eps
        return x, key

    def em_body(i, carry):
        # Forward SDE: dX = -gamma X dt + sqrt(2 gamma) dW.
        # Reverse EM with dt>0 stepping backward (Anderson / Song).
        x, key = carry
        key, kn = jax.random.split(key)
        t = ts[i]
        t_next = ts[i + 1]
        dt = t - t_next
        score = apply_fn(params, x, z_b, t)
        g2 = 2.0 * gamma
        g = jnp.sqrt(g2)
        f_x = -gamma * x
        x_mean = x - (f_x - g2 * score) * dt
        x = x_mean + g * jnp.sqrt(dt) * jax.random.normal(kn, x.shape, dtype=jnp.float32)
        return x, key

    body = ddpm_body if method == "ddpm" else em_body
    x, _ = jax.lax.fori_loop(0, n_steps, body, (x, key))
    return x
