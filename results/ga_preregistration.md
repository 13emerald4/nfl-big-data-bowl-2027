# Genetic-algorithm search for a receiver cut metric, and athlete rankings: analysis plan

Written before any of the new metrics was extracted or related to an outcome.
Already known (and therefore not evidence for this search): Stage 3 found that
cut speed at the last cut before the throw (`apex_z`) raises separation within
receiver (+0.12 yd per SD, 2023–25 pooled).

`scripts/ga_cut_features.py` extracts the metrics; `scripts/metric_search_ga.py`
runs the search, the null test, the 2025 holdout and the rankings.

## Who is ranked

Cohort wide receivers in NFL regular-season games (game tracking), on pass
plays where they ran a route and made at least one cut before the throw. The
sample filters are those of Stage 3 (`play_level_fe.py`). Ranked: receivers with
at least 30 such plays over 2023–25.

## Metrics (per play, at the last cut before the throw)

1. **Cut speed**: speed at the slowest frame of the cut (m/s).
2. **Speed kept**: cut speed ÷ peak speed in the 1.0 s before the cut.
3. **Re-acceleration**: mean tangential acceleration over the 0.5 s after the cut.
4. **Recovery time**: time from the cut until speed is back to 90% of entry speed
   (capped at 1.5 s), sign flipped so higher is better.
5. **Braking**: peak deceleration within ±window of the cut (m/s²).
6. **Turning acceleration**: peak normal acceleration within ±window (m/s²).

Accelerations are per unit mass (force ÷ mass), so they are already size-relative.

**Size metrics, for ranking only:** momentum through the cut (Combine mass in kg ×
the receiver's mean cut speed, kg·m/s) and speed for size (the receiver's mean
cut speed minus the value expected for his weight, OLS across receivers). Mass is
constant within a player, so within-player tests cannot validate these. They
are ranked descriptively and are not part of the search.

## What the genetic algorithm searches over

- Cut detection: minimum turn angle {30, 45, 60}° × window {±0.3, ±0.5, ±0.7} s
  (nine settings; smoothing fixed at the validated 7-frame Savitzky–Golay filter).
- Peer group for z-scoring every metric: position, route, or route × turn-angle bin.
- Which metrics enter (1 to 4) and their weights (each in [−1, 1]).
- The composite is the weighted sum of the z-scored metrics, standardised.
  The player score is the player's mean composite.

## Objectives (training data: 2023 and 2024 seasons only)

1. **Validity (maximise):** within-receiver effect of the composite on separation
   at the throw, yards per SD, from the Stage 3 model (player, route, coverage
   type, alignment, down and opponent × season fixed effects; turn angle, cut
   time, yards to go, cushion and dropback covariates).
2. **Reliability (constraint ≥ 0.60):** split-half reliability of the player
   score (random halves of each player's plays, Spearman–Brown), receivers with
   at least 10 plays per half.
3. **Simplicity (minimise):** number of metrics.

NSGA-II, population 80, 40 generations, seed 2027.

**Selection rule:** among feasible frontier members, take those within 90% of the
best validity, then the fewest metrics, then the highest validity.

## Tests

- **Null (search overfitting):** rerun the whole search 200 times with
  separation residuals (from the model without the composite) permuted within
  receiver (Freedman–Lane), applying the same selection rule. p = share of null
  runs whose selected validity reaches the observed one.
- **Holdout (primary): the selected composite on the 2025 season,** untouched
  until the final run. Within-receiver effect with wild cluster bootstrap p and
  interval. z-scoring and standardisation use training-season statistics.
- **Benchmark:** the Stage 3 metric alone (cut speed, 45°, ±0.5 s, route ×
  angle z-score) on 2025. The search adds value only if the composite's 2025
  effect exceeds it.

## Rankings

- Player score on all seasons; 2,000 bootstrap resamples of each player's plays
  give a 95% interval for his rank. Reported as rank intervals.
- Descriptive between-player checks (not tests of the search): Spearman
  correlation of the player score with the player's mean separation and with
  yards per route run.
- Momentum and speed-for-size rankings with bootstrap rank intervals.
