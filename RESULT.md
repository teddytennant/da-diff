# RESULT

## Commands

```
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e ".[tests]"
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -v --tb=line
JAX_PLATFORMS=cpu .venv/bin/python train.py
```

## Tests

9 passed, 0 failed, ~18s CPU.

```
tests/test_cit.py::test_pvalue_formula_canned PASSED
tests/test_cit.py::test_pvalue_at_least_one_over_one_plus_b PASSED
tests/test_cit.py::test_cit_null_not_tiny PASSED
tests/test_cit.py::test_cit_alternative_smaller_than_null PASSED
tests/test_da_diff.py::test_dsm_loss_decreases PASSED
tests/test_da_diff.py::test_weighted_erm_warmup_and_weights PASSED
tests/test_da_diff.py::test_reverse_sample_shape PASSED
tests/test_da_diff.py::test_forward_noising_t1_near_prior PASSED
tests/test_da_diff.py::test_source_weights_nonnegative PASSED
```

Demo (`train.py`): last losses finite; source weights nonnegative; null-ish p ~ 0.94; alternative p ~ 0.059.

## Mismatches vs paper

- Weight solver is a closed-form DSM transfer-bias proxy, not MM-QP / gradient alignment (Appendix C).
- Tiny 2-layer ReLU MLP with Fourier time features, not the paper architecture.
- Unit-interval log-linear VP (gamma=4) instead of OU on a long [0, T].
- 10--50 DDPM-style reverse steps, not a long reverse SDE.
- T is partial dCov (OLS residuals then biased dCov^2); paper uses dCor(X, Y, Z).
- No D0 train/test split in the CIT API.
- No Sachs / real data, no W2 tables, Type I bound not reproduced at paper scale.
