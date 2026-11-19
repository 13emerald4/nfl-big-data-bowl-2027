# Combine-to-game transfer: analysis plan

Written before any in-game cut metric was computed. `scripts/game_cod_features.py`
builds the game metrics; `scripts/transfer_test.py` runs this plan.

## Question

Do the Combine change-of-direction metrics (`scripts/cod_features.py`) measure
what players do in games? If a Combine metric does not track its in-game
counterpart, it cannot be the mechanism behind any game outcome, and the
Combine null results say nothing about the in-game skill.

## In-game metrics

Regular-season game tracking (10 Hz), every cohort player. Each player-play is
smoothed and cut-detected exactly as at the Combine (Savitzky-Golay, 7 frames;
cut = slowest frame of a run turning >= 45 degrees over +/- 0.5 s). Only cuts
from the snap to the throw (pass plays; at most 4 s) or the first 3 s after
the snap (other plays) count.

| Name | Mirrors | Definition |
|:--|:--|:--|
| `g_route_speed_z` | `cod_speed_z` | Receivers on routes: apex speed z-scored within route x turn-angle bin, mean per player |
| `g_cut_speed_z` | `tc_cut_speed_z` | Every position: apex speed z-scored within position group x turn-angle bin, mean per player |
| `g_force_oe_z` | `force_oe_z` | Median peak horizontal force per cut, residualised on body mass, z-scored |
| `g_turn_oe_z` | `turn_oe_z` | Same, turning (normal) component |
| `g_brake_oe_z` | `brake_oe_z` | Same, braking (tangential) component |

Players need >= 30 in-game cuts.

## Tests

Transfer is the Pearson correlation between each Combine metric and its
in-game counterpart, with a 95% Fisher interval. Force metrics are also
reported as partial correlations given position group, because role shapes
in-game cuts. Each correlation is also reported disattenuated,
r / sqrt(rel_combine x rel_game), using split-half reliabilities (Combine:
`results/cod_reliability.txt`; game: odd vs even plays, Spearman-Brown).

**Primary (Holm across the three):**
1. `cod_speed_z` vs `g_route_speed_z` (receivers).
2. `force_oe_z` vs `g_force_oe_z` (all positions, partial on position group).
3. `turn_oe_z` vs `g_turn_oe_z` (all positions, partial on position group); added
   because turning force was the only exploratory lead in the Combine models.

## Game metrics and outcomes (exploratory)

In-game metrics and outcomes from the same games share role, scheme and
playing time, so the outcome test is temporal: the metric from a player's
first season predicts yards per route run in his later seasons (>= 100 routes
each side; 2023 and 2024 classes), with draft controls and HC1 errors.
Contemporaneous associations are shown for reference only.
