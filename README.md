JAX kernel of domain-adapted diffusion (DA-Diff) and the CRT conditional independence test (DA-CIT) from arXiv:2609.33490.

## Install

```
python3 -m venv .venv
.venv/bin/pip install -e ".[tests]"
```

## Run

```
.venv/bin/python -m pytest -q
.venv/bin/python train.py
```

## What does not match the paper

No MM-QP / gradient-alignment weight solver (Appendix C, Alg. 1 steps 10--13): after warmup, source weights are a closed-form surrogate w_k proportional to n_k / (|L_k^{DSM} - L_0^{DSM}| + eps), renormalized to mean 1, optionally refreshed every update_every epochs, and frozen between updates. The score network is a tiny ReLU MLP with Gaussian Fourier time features, not the paper's architecture. The reverse process uses 10--50 DDPM-style (or Euler--Maruyama) steps on a unit-interval log-linear VP schedule, not the paper's long OU horizon. T is biased distance covariance of OLS residuals of X and Y on Z (partial dCov), with unconditional dCov available; this is not necessarily the paper's exact dCor(X, Y, Z). No Sachs or other real data, no W2 tables, and the Type I bound is not empirically reproduced at paper scale.
