# Change-of-direction speed and force: analysis plan

Written after the metrics were built from drill data (`scripts/cod_features.py`,
`results/cod_reliability.txt`) and **before any career or game outcome was
read for these metrics**. `scripts/cod_force_models.py` runs exactly this plan.

Caveat: the same 2023-2025 outcomes were analysed earlier with related metrics
(pooled cut speed; drill-standardised cut speed z_cut, p = 0.058 on yards per
route run). This plan limits, but cannot remove, that prior look.

## Claims the design can support

The data are observational. A significant result means "the metric predicts
the outcome among players drafted at the same slot". Reading it as a cause
requires assuming no unmeasured confounder (route skill, scheme, quarterback,
injury). Draft pick may lie on the causal path (teams draft on athleticism),
so each model is also reported without draft controls (total association).

## Metrics (frozen)

| Name | Idea | Definition |
|:--|:-:|:--|
| `cod_speed_z` | 1 | Receiver route drills: speed at the slowest frame of each cut (turn >= 45 degrees), z-scored within drill x turn-angle bin, averaged within drill, then across drills (>= 3 drills) |
| `cod_retention_z` | 1 | Same, for apex speed / entry speed |
| `cod_exit_z` | 1 | Same, for speed 0.5 s after the apex |
| `tc_cut_speed_z` | 1 | Three-cone, every position: mean apex speed through the rounded 45-90 degree cuts, z-scored within position group |
| `force_oe_z` | 2 | Shuttle and three-cone: median per rep of the peak horizontal force F = m\|a\| within 0.5 s of each cut, residualised on body mass within drill, z-scored, averaged across drills |
| `force_per_kg_z` | 2 | Peak force / body mass, z-scored within drill |
| `force_abs_z` | 2 | Peak force in newtons, z-scored within drill |
| `brake_oe_z`, `turn_oe_z` | 2 | Braking (tangential) and turning (normal) components, residualised on mass |
| `fv_f0`, `fv_slope` | 2 | Linear fit of 90th-percentile force on speed across 1 m/s bands (force-velocity profile) |

## Outcomes

- **YPRR**: regular-season receiving yards per route run, receivers with >= 100 routes.
- **Snaps per season**: career offensive + defensive snaps, with seasons of exposure as the offset.
- **Starter**: >= 10 career starts.
- Targets per route, separation at the throw (receivers); pass-rush pressure rate (DL, >= 100 rushes).

## Controls

`log(pick) + UDFA + draft-class fixed effects`, plus position-group fixed
effects for multi-position samples. Idea 2 adds body mass where the metric is
not already mass-adjusted (`force_abs_z`, `force_per_kg_z`).

## Primary tests (two; Holm across both)

1. **Idea 1:** `cod_speed_z` -> YPRR, OLS with HC1 standard errors.
2. **Idea 2:** `force_oe_z` -> snaps per season, Poisson QMLE with robust standard errors.

Each is reported with a 95% confidence interval and in plain units. Everything
else is exploratory and is Holm-corrected as one family.

## Conformal prediction

Jackknife+ 80% prediction intervals for each player's outcome from a
draft-controls model, with and without the metric. Reported: empirical
coverage, mean interval width, and the width change. A permutation test
(the metric shuffled 1,000 times) gives the probability that a metric of pure
noise narrows the intervals as much.

## Known weaknesses, stated in advance

- Reliability (`results/cod_reliability.txt`): cut-speed z agrees 0.58 between
  two reps of the same drill but 0.03 between different drills (split-half
  reliability 0.20). Force over expected agrees 0.38 rep to rep and 0.14
  between shuttle and three-cone. Low reliability biases every effect toward
  zero and widens intervals.
- Horizontal force only; the tracked point is not the centre of mass; 10 Hz
  positions differentiated twice.
- The 2025 class has one season of outcomes.
