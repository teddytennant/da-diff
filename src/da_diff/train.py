"""DA-Diff training: warmup on the target then weighted ERM (Alg. 1, eqs. 10, 13).

Weight selection is a closed-form surrogate, not the paper's MM-QP /
gradient-alignment solver (Appendix C). After warmup, per-domain DSM losses
of the current score give a transfer-bias proxy
    w_k proportional to n_k / (|L_k - L_0| + eps),
renormalized to mean 1. Weights stay nonnegative. W_N = sum_k w_k n_k.
Optionally re-estimated every `update_every` epochs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from da_diff.score import apply_score, init_score
from da_diff.sde import GAMMA, T0, alpha_sigma, dsm_target, forward_noise


def _adam_init(params, lr):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32), "lr": lr}


def _adam_update(params, grads, state, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    lr = state["lr"]
    m = jax.tree_util.tree_map(lambda mm, g: b1 * mm + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda vv, g: b2 * vv + (1.0 - b2) * g * g, state["v"], grads)
    mhat = jax.tree_util.tree_map(lambda mm: mm / (1.0 - b1 ** t), m)
    vhat = jax.tree_util.tree_map(lambda vv: vv / (1.0 - b2 ** t), v)
    params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat
    )
    return params, {"m": m, "v": v, "t": t, "lr": lr}


def per_sample_dsm(params, key, x, z, t0: float = T0, gamma: float = GAMMA):
    """Monte Carlo DSM (eq. 6) per row. x,z: (n, d)."""
    x = jnp.asarray(x, dtype=jnp.float32)
    z = jnp.asarray(z, dtype=jnp.float32)
    n = x.shape[0]
    k_t, k_e = jax.random.split(key)
    t = jax.random.uniform(k_t, (n,), minval=t0, maxval=1.0, dtype=jnp.float32)
    xt, eps = forward_noise(k_e, x, t, gamma=gamma)
    sigma = alpha_sigma(t, gamma=gamma)[1][:, None]
    score = apply_score(params, xt, z, t)
    target = dsm_target(eps, sigma)
    return jnp.sum((score - target) ** 2, axis=-1)


def dsm_loss(params, key, x, z, t0: float = T0, gamma: float = GAMMA):
    """Mean DSM on a single domain (eqs. 6--7, 9)."""
    return jnp.mean(per_sample_dsm(params, key, x, z, t0=t0, gamma=gamma))


def weighted_dsm_loss(
    params,
    key,
    x0,
    z0,
    sources,
    weights,
    t0: float = T0,
    gamma: float = GAMMA,
):
    """Weighted ERM (eq. 10):

        L_w = (n0 L0 + sum_k w_k n_k L_k) / (n0 + W_N),  W_N = sum_k w_k n_k.
    """
    weights = jnp.asarray(weights, dtype=jnp.float32)
    n0 = x0.shape[0]
    keys = jax.random.split(key, 1 + len(sources))
    ell0 = per_sample_dsm(params, keys[0], x0, z0, t0=t0, gamma=gamma)
    num = jnp.sum(ell0)
    den = jnp.asarray(n0, dtype=jnp.float32)
    lk = []
    ns = []
    for k, (xk, zk) in enumerate(sources):
        nk = xk.shape[0]
        ellk = per_sample_dsm(params, keys[k + 1], xk, zk, t0=t0, gamma=gamma)
        wk = weights[k]
        num = num + wk * jnp.sum(ellk)
        den = den + wk * nk
        lk.append(jnp.mean(ellk))
        ns.append(nk)
    loss = num / jnp.clip(den, 1e-8)
    ns = jnp.asarray(ns, dtype=jnp.float32)
    W_N = jnp.sum(weights * ns)
    aux = {
        "W_N": W_N,
        "weights": weights,
        "L0": jnp.mean(ell0),
        "Lk": jnp.stack(lk) if lk else jnp.zeros((0,), dtype=jnp.float32),
        "n0": jnp.asarray(n0, dtype=jnp.float32),
        "ns": ns,
    }
    return loss, aux


def source_weights_from_dsm(L0, Ls, ns, eps: float = 1e-3):
    """Closed-form surrogate for eq. 13: w_k ∝ n_k / (|L_k - L_0| + eps).

    Renormalized so mean(w)=1. Not MM-QP / gradient alignment.
    """
    Ls = jnp.asarray(Ls, dtype=jnp.float32)
    ns = jnp.asarray(ns, dtype=jnp.float32)
    L0 = jnp.asarray(L0, dtype=jnp.float32)
    bias = jnp.abs(Ls - L0) + eps
    w = ns / bias
    w = w / jnp.clip(jnp.mean(w), 1e-8)
    return jnp.clip(w, 0.0, None)


def _domain_losses(params, key, x0, z0, sources, t0, gamma):
    keys = jax.random.split(key, 1 + len(sources))
    L0 = dsm_loss(params, keys[0], x0, z0, t0=t0, gamma=gamma)
    Ls = []
    ns = []
    for k, (xk, zk) in enumerate(sources):
        Ls.append(dsm_loss(params, keys[k + 1], xk, zk, t0=t0, gamma=gamma))
        ns.append(xk.shape[0])
    Ls = jnp.stack(Ls) if Ls else jnp.zeros((0,), dtype=jnp.float32)
    ns = jnp.asarray(ns, dtype=jnp.float32)
    return L0, Ls, ns


def train_da_diff(
    key,
    x0,
    z0,
    sources,
    *,
    hidden: int = 64,
    epochs: int = 20,
    warmup_epochs: int = 5,
    lr: float = 1e-3,
    update_every: int = 5,
    t0: float = T0,
    gamma: float = GAMMA,
    weight_mode: str = "proxy",
    time_dim: int = 16,
):
    """Algorithm 1: ew warmup epochs on L0, then weighted ERM L_w.

    sources: list of (xk, zk). weight_mode: "proxy" (default) or "equal".
    """
    x0 = jnp.asarray(x0, dtype=jnp.float32)
    z0 = jnp.asarray(z0, dtype=jnp.float32)
    sources = [(jnp.asarray(x, dtype=jnp.float32), jnp.asarray(z, dtype=jnp.float32)) for x, z in sources]
    dx, dz = x0.shape[-1], z0.shape[-1]
    key, k_init = jax.random.split(key)
    params = init_score(k_init, dx, dz, hidden=hidden, time_dim=time_dim)
    opt = _adam_init(params, lr)
    K = len(sources)
    weights = jnp.ones((K,), dtype=jnp.float32) if K else jnp.zeros((0,), dtype=jnp.float32)

    def loss_target(p, k):
        return dsm_loss(p, k, x0, z0, t0=t0, gamma=gamma)

    def loss_weighted(p, k, w):
        loss, aux = weighted_dsm_loss(p, k, x0, z0, sources, w, t0=t0, gamma=gamma)
        return loss, aux

    history = []
    W_N = jnp.asarray(0.0, dtype=jnp.float32)

    for e in range(epochs):
        key, k_step, k_w = jax.random.split(key, 3)
        if e < warmup_epochs or K == 0:
            loss, grads = jax.value_and_grad(loss_target)(params, k_step)
            aux = {"W_N": jnp.asarray(0.0, dtype=jnp.float32), "weights": weights}
        else:
            if (e - warmup_epochs) % update_every == 0:
                if weight_mode == "equal":
                    weights = jnp.ones((K,), dtype=jnp.float32)
                else:
                    L0, Ls, ns = _domain_losses(params, k_w, x0, z0, sources, t0, gamma)
                    weights = source_weights_from_dsm(L0, Ls, ns)
            (loss, aux), grads = jax.value_and_grad(loss_weighted, has_aux=True)(
                params, k_step, weights
            )
            W_N = aux["W_N"]
        params, opt = _adam_update(params, grads, opt)
        history.append(float(loss))

    return {
        "params": params,
        "weights": weights,
        "W_N": W_N,
        "history": history,
        "apply_fn": apply_score,
        "dx": dx,
        "dz": dz,
    }
