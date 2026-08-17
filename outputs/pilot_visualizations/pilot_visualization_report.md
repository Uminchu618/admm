# Pilot simulation visualization report

- Source: `outputs/pilot_summary.csv`
- Rows: 1,200
- Datasets: 100
- Lambda candidates: 12
- Residual screening rule: primal and dual residual <= 0.01
- This screen is deliberately conservative and is not a replacement for the solver's recorded stopping rule.

## Main diagnostic finding

- Positive-lambda fits passing the residual screen: 0/1100.
- BIC-selected positive-lambda fits passing the residual screen: 0/91.
- BIC selected a positive lambda in 91/100 datasets.
- Therefore, positive-lambda BIC results should not be interpreted as valid fused-lasso estimates until convergence is fixed or formally verified from result.json.

## BIC-selected descriptive summary

| Scenario | n | Median lambda | Mean test Ctd | Mean change points | Median primal residual | Screen pass rate |
|---|---:|---:|---:|---:|---:|---:|
| Oracle | 20 | 0.25 | 0.7011 | 4.50 | 0.1897 | 30.0% |
| Fine-grid | 20 | 0.25 | 0.7015 | 3.30 | 0.3955 | 10.0% |
| Off-grid | 20 | 0.25 | 0.6942 | 0.00 | 0.7019 | 0.0% |
| Small | 20 | 0.25 | 0.6837 | 0.00 | 0.5948 | 0.0% |
| No-change | 20 | 0.25 | 0.6677 | 1.65 | 0.6382 | 5.0% |

## Scope limitation

The summary CSV does not contain coefficient-function error, matched change-point precision/recall, or localization error. Those require the per-fit result.json files and the scenario truth definitions.
