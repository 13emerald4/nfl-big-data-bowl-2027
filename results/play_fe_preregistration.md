# Play-level fixed-effects test: analysis plan

Written before the play-level dataset was built. `scripts/play_level_fe.py`
runs this plan on `data/game_cuts.parquet` (from `scripts/game_cod_features.py`)
and `player_play.csv`.

## Question

On plays where a receiver holds more speed through his cut than he usually
does (idea 1), or produces more turning force through it (idea 2), is he
more open when the ball is thrown?

Each receiver is compared with himself (player fixed effects), so draft slot,
talent, size and team drop out. This is the closest the data get to "A causes
B"; what remains is confounding that varies from play to play.

## Sample

Regular-season pass plays on which a cohort wide receiver ran a route and had
at least one detected cut (turn >= 45 degrees) between the snap and the throw.
Separation at the throw is recorded on every targeted route and on 85% of
untargeted ones, so the outcome is not limited to plays the quarterback chose.
Receivers need >= 30 such plays.

## Treatment (the last cut before the throw)

- Idea 1: `apex_z`, apex speed z-scored within route x turn-angle bin.
- Idea 2: `turn_force`, peak turning (normal) force within 0.5 s of the cut,
  per 100 N. Body mass is constant within player, so within-player variation
  in force is variation in turning acceleration.

## Outcomes

- **Primary:** separation at the throw (yards).
- Secondary: targeted (linear probability model).

## Model

OLS with player fixed effects and fixed effects for route, coverage type,
receiver alignment, down, and opponent x season; covariates turn angle, cut
time after the snap, yards to go, pre-snap cushion, and the play's dropback
duration. Standard errors clustered by player. Effects are per 1 SD of apex_z
and per 100 N of turning force, with 95% confidence intervals.

## Tests

**Primary (Holm across the two):** apex_z -> separation; turn_force -> separation.

Pre-specified secondary: each treatment in man and zone coverage separately
(a receiver who sees he is open in zone may throttle down and sit, which would
make the zone estimate negative for reasons other than cutting skill, so man
coverage is the cleaner test); both treatments in one model; target as the
outcome; tight ends instead of receivers. Holm across the secondary family.

Conformal: 10-fold CV+ 80% prediction intervals for separation from the
fixed-effects model with and without the treatment; width change and a
200-shuffle permutation test.

## Known weaknesses

- Play design: a route called to be run sharper may also be called against a
  coverage it beats; route and coverage fixed effects absorb only part of this.
- The defender's movement is not tracked in this dataset (cohort players only).
- Reverse causation in zone coverage (see above).
- Throw timing: separation is measured at the throw, which the quarterback
  times partly on how open the receiver looks.
