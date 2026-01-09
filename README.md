## Module Layout

```
admm/
  __init__.py        public exports
  model.py           ADMMHazardAFT estimator + fit flow
  solver.py          ADMM solver (fused lasso)
  objective.py       log-likelihood + gradient + hessian
  baseline.py        baseline hazard models (B-spline, etc.)
  time_partition.py  time grid + interval utilities
  quadrature.py      numerical integration rule
  config.py          config loader (TOML/JSON)
  types.py           shared type aliases
main.py              CLI entrypoint (loads config)
config.toml          example config
```
