"""DA-Diff + DA-CIT: domain-adapted diffusion for conditional independence testing."""

from da_diff._host import preload as _preload

_preload()

from da_diff.cit import (
    crt_pvalue,
    da_cit,
    dependence_stat,
    distance_covariance,
    partial_distance_covariance,
)
from da_diff.score import apply_score, init_score
from da_diff.sde import GAMMA, T0, alpha_sigma, dsm_target, forward_noise, reverse_sample
from da_diff.train import (
    dsm_loss,
    source_weights_from_dsm,
    train_da_diff,
    weighted_dsm_loss,
)

__all__ = [
    "GAMMA",
    "T0",
    "alpha_sigma",
    "apply_score",
    "crt_pvalue",
    "da_cit",
    "dependence_stat",
    "distance_covariance",
    "dsm_loss",
    "dsm_target",
    "forward_noise",
    "init_score",
    "partial_distance_covariance",
    "reverse_sample",
    "source_weights_from_dsm",
    "train_da_diff",
    "weighted_dsm_loss",
]
