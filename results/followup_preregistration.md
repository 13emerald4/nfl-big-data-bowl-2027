# Follow-up causal tests: analysis plan

Written before any of these treatments was extracted or related to an outcome.
Up to this point only field coverage was inspected (row counts, value ranges,
event names, and the fact that `o` and `dir` share a convention). No
treatment–outcome association had been computed.

`scripts/followup_features.py` builds the treatments from game tracking;
`scripts/followup_tests.py` runs this plan.

## Common rules

- Regular-season plays only; plays nullified by a penalty are dropped.
- Every model is OLS (linear probability for binary outcomes) with the unit's
  own fixed effect, so each player is compared with himself.
- Inference: CR1 standard errors clustered as stated, plus a restricted wild
  cluster bootstrap (Rademacher weights, 9,999 draws) for the p-value and a
  95% test-inversion interval. These are the reported numbers.
- Each test has one primary estimate. **Holm across the seven primaries.**
  Secondary estimates are labelled as such; Holm within each test's secondary set.
- Continuous treatments are reported per 1 SD of their within-player variation
  (and in natural units).
- A test whose sample falls below its minimum is reported as descriptive only.

## T1. Mediation: cut speed → separation → expected points

Sample: the Stage 3 receiver sample (`data/play_level.parquet`, WR).
Outcome: receiver EPA = the play's EPA if the receiver was targeted, else 0.
Treatment: `apex_z` (speed at the last cut before the throw). Mediator: separation.
Model: the Stage 3 fixed effects and covariates.

- a = effect of apex_z on separation; b = effect of separation on receiver EPA,
  holding apex_z; **primary: indirect effect a·b.**
- Inference for a·b: cluster bootstrap over receivers (4,999 draws), percentile 95% interval
  and two-sided p (share of draws on the other side of 0, doubled).
- Secondary: total effect of apex_z on receiver EPA; direct effect; effect on
  target probability and on completion given a target.
- Assumption stated with the result: no unmeasured play-level confounder of
  separation and EPA (sequential ignorability). The indirect effect is
  therefore a decomposition under that assumption, not a separate experiment.

## T2. Speed at the catch → yards after the catch

Sample: catches by cohort WR, TE and RB with tracking at `pass_outcome_caught`
and a recorded YAC. Minimum 300 catches and 20 players.
Treatment: speed (m/s, provider `s`) at the `pass_outcome_caught` frame.
Fixed effects: player, route, coverage type, down, opponent × season;
covariates: air yards (from `pass_length`), yards to go, field position
(distance to the end zone). Clustered by player.

- **Primary:** YAC (yards).
- Secondary: YAC over expected (YAC − `expected_yards_after_catch`); P(YAC ≥ 5).
- Caveat stated in advance: the provider's expected-YAC model probably uses
  speed at the catch, so the YACOE estimate asks only whether speed adds
  beyond that model. A null there is not evidence against the primary.

## T3. Pass-rusher get-off → pressure

Sample: cohort rushers with `player_get_off` (EDGE and interior). Minimum 30
rushes per player.
Treatment: `player_get_off` (seconds; lower = faster), sign flipped so a
positive coefficient means a faster get-off helps.
Fixed effects: player, alignment (`lined_up_position`), down, opponent × season;
covariates: yards to go, blitz flag. Clustered by player.

- **Primary:** pressure (1 if `time_to_pressure` is recorded or a sack is credited).
- Secondary: sack; time to pressure on pressured plays; adding the play's
  dropback duration (a possible mediator, so not in the primary).

## T4. Offensive-line get-off → pressure allowed

Sample: cohort linemen (G, T, C) on pass plays with `peak_pressure_probability_allowed`.
Minimum 30 snaps per player.
Treatment: OL get-off = time from the snap until the lineman's speed first
reaches 1.0 m/s (capped at 2.0 s), sign flipped as in T3.
Fixed effects: player, alignment, down, opponent × season; covariate: yards to go.
Clustered by player.

- **Primary:** peak pressure probability allowed (provider model, 0–1).
- Secondary: pressure allowed (binary); set depth (yards retreated by 1.0 s,
  per yard) as an alternative treatment.

## T5. Shoulders before the cut (selling the route) → separation

Sample: the Stage 3 receiver sample.
Treatment: early shoulder turn = signed angle (degrees) between the receiver's
orientation `o` 0.3 s before the cut and his travel direction over the stem
(0.5–1.0 s before the cut), positive when the shoulders have turned toward
the coming break. z-scored within route × turn-angle bin like apex_z.
Model: the Stage 3 model. Clustered by receiver.

- **Primary:** separation on early shoulder turn (total effect; two-sided).
- Secondary: adding apex_z (effect beyond speed); man vs zone.

## T6. Head-to-head speed at the cut

Sample: Stage 3 receiver plays in man coverage on which a cohort defender with
`coverage_assignment == "MAN"` is tracked. The defender covering the receiver
is the one with the smallest mean distance to him from the snap to the cut,
kept only if within 3 yards at the cut. Minimum 200 plays and 20 receivers.
Treatment: speed gap = receiver speed − defender speed at the receiver's cut
frame (m/s).
Model: separation on the gap, receiver fixed effects, Stage 3 covariates,
coverage type, opponent × season dropped if the sample cannot support it
(decided by the minimum-cell rule of Stage 3, before results). Clustered by receiver.

- **Primary:** speed gap → separation.
- Secondary: receiver and defender speed entered separately; defender fixed
  effects instead of receiver fixed effects (the defensive mirror).

## T7. Days of rest → game speed

Sample: cohort player-games, weeks 2–18, with at least 15 tracked plays.
Treatment: days since the team's previous game, from `game_id` dates:
short (≤ 5 days, mostly Thursday games), normal (6–8), long (≥ 9, mostly after a bye).
Outcome: the player's peak speed in the game (maximum `s`, km/h).
Fixed effects: player, season × week (accumulated fatigue and calendar),
home game. Clustered by team-season (rest is set per team game).

- **Primary:** short vs normal rest → peak speed, skill positions (WR, TE, RB, CB, S).
- Secondary: long vs normal; all positions; the receiver's mean apex_z in the
  game as the outcome.
- Rest is set by the league schedule, so it is close to random for a given
  player; the threat is that teams manage snaps on short weeks, which the
  play-count minimum and a snaps covariate (secondary) address.
