# Beyond the Stopwatch, Behind the Draft Board: What Combine Tracking Adds Once Draft Position Is Known

[![NFL Big Data Bowl 2027](https://img.shields.io/badge/NFL%20Big%20Data%20Bowl-2027-013369.svg)](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Submission for the NFL Big Data Bowl 2027 (Kaggle Analytics Track)**  
> **Author:** Michael Polley  
> **Theme:** Linking 10 Hz Combine sensor tracking to regular-season NFL performance

---

## Summary

Combine tracking metrics correlate with NFL playing time, but almost entirely because teams already draft on them. Every model is fit twice: on the Combine metric alone, and again with draft pick, an undrafted flag and draft-class (and position) fixed effects.

| Metric → outcome | Metric alone | + draft controls |
|:--|:-:|:-:|
| WR cut speed → career snaps (per km/h) | +114, p = 0.002 | −10, p = 0.74 |
| WR cut speed → starter (odds ratio) | 1.63, p = 0.026 | 1.11, p = 0.76 |
| DL mass-adjusted acceleration (γ_OE) → starter (odds ratio) | 1.85, p = 0.042 | 1.52, p = 0.35 |
| DB break speed → career snaps | +192, p = 0.011 | −15, p = 0.80 |
| OL γ_OE → career snaps | +155, p = 0.015 | +58, p = 0.32 |

* Draft variables explain **51%** of receiver snap variance. Cut speed and the 40 time add **0.2 points**.
* Of 22 controlled tests, one has p < 0.05, about what chance predicts. **None survives Holm correction.**
* Two per-snap efficiency leads remain near p = 0.06:
  * drill-standardised cut speed → yards per route run (+0.32 per unit, p = 0.058);
  * pass-rush-drill γ_OE → starter (odds ratio 1.60, p = 0.059).
  
  They are candidates for a pre-registered test on the 2026 class, not draft rules.
* **Speed through the cut, in three pre-registered stages** (`results/*_preregistration.md`):
  * **Combine → careers:** cut speed → YPRR +0.35 per SD (p = 0.18); size-adjusted force → snaps (p = 0.17). Not significant.
  * **Combine → games:** the Combine metrics do not track the same quantities measured in 166,150 in-game cuts (cut speed r = +0.02, force r = −0.16). Combine cut speed agrees only 0.03 across drills.
  * **Within player, in games:** holding speed through the last cut before the throw goes with **+0.12 yd of separation per SD** (Holm p = 0.002; +0.19 vs man coverage; +3 points target rate), comparing each receiver with himself over 4,450 routes. Turning force is not the lever.
  * The skill is real, but Combine drills don't measure it.
  * **Checks that don't rely on normal approximations** (`scripts/distribution_checks.py`):
    * A wild cluster bootstrap gives CI [+0.05, +0.19], p = 0.003.
    * Within-receiver Freedman–Lane permutation gives p = 0.003.
    * Spearman transfer correlations: ρ = +0.01 and −0.18.
  * **Where the gain sits:** P(separation > 1, 2, 3 yd) rises by 3.6–4.6 points per SD, while P(> 4 yd) is unchanged. Holding speed turns tight coverage into open targets, not wide-open ones.
* **Seven follow-up tests, pre-registered** (`results/followup_preregistration.md`; paper Appendix B). Each compares a player with himself, with wild-bootstrap p-values and Holm correction:
  * **Cut speed → EPA:** through separation, +0.006 EPA per SD per route (Holm p = 0.005). The total effect is not significant.
  * **Speed at the catch → YAC:** +0.83 yd per m/s (p < 0.001). It holds beyond expected YAC and with separation controlled.
  * **Pass-rush get-off → pressure:** +0.47 points per 0.1 s faster (p < 0.001); +0.68 on pass plays only (post hoc).
  * **Lineman get-off → pressure allowed:** significant but the **wrong sign**, most likely the lineman reacting to a fast rusher. Not a lever.
  * **Nulls:** shoulders turned before the cut → separation (p = 0.48); short rest → peak speed (+0.16 km/h, CI rules out a drop > 0.1).
  * **Head-to-head speed gap vs the man defender:** only 156 plays, below the pre-set minimum. Descriptive only.
* **Predictor ranking** (`scripts/rank_predictors.py`):
  * **Playing time:** draft pick ranks first at every position (starter AUC 0.82–0.93), then the NGS final and college-production scores. Nothing adds to draft pick out of sample.
  * **Per-snap efficiency:** the strongest signals beyond draft slot are drill-standardised cut speed for receiver YPRR (+0.33, p = 0.008), and the NGS final score (+0.36) and 10-yard split (−0.29, p = 0.034) for pass-rush pressure rate.
  * Exploratory: none of the 30 efficiency correlations survives Holm correction.

---

## Artifacts

| Artifact | Description |
| :--- | :--- |
| [`REPORT.md`](./REPORT.md) / [`REPORT.html`](./REPORT.html) / [`REPORT.pdf`](./REPORT.pdf) | Competition write-up as a research paper with abstract (< 2,000 words excluding references and appendix, 5 figures + 1 appendix figure, 4 tables) |
| [`report_assigned.pdf`](./report_assigned.pdf) | 4-page formal report (LaTeX) |
| [`presentation.pdf`](./presentation.pdf) | 10-slide Beamer deck |
| [`draft_controls.png`](./draft_controls.png) | Test statistic for every model, before vs after draft controls |
| [`wr_speed_pdf_cdf.png`](./wr_speed_pdf_cdf.png) | Cutting-drill speed density and CDF for four case-study receivers |
| [`route_kinematics_field_plot.png`](./route_kinematics_field_plot.png) | Comeback route: Nacua vs Johnston paths and speed through the break |
| [`mass_vs_gamma_scatter.png`](./mass_vs_gamma_scatter.png) | Defensive-line weight vs 90th-percentile acceleration |
| [`play_level_cut_speed.png`](./play_level_cut_speed.png) | Within-player effect of speed through the cut on separation |
| [`last_cut_speed_distribution.png`](./last_cut_speed_distribution.png) | Receivers' speed at the last cut before the throw: PDF by turn angle, CDF by open vs tight (1 SD ≈ 4.4 km/h) |
| [`separation_distribution.png`](./separation_distribution.png) | Adjusted separation PDF/CDF by the receiver's own cut speed, and effects on P(separation > c) |
| [`combine_vs_game_cuts.png`](./combine_vs_game_cuts.png) | CDFs of cut speed and turn angle for the same receivers, Combine vs games |
| [`permutation_nulls.png`](./permutation_nulls.png) | Within-receiver permutation null distributions (appendix) |
| [`tier_comparison_pdf_cdf.html`](./tier_comparison_pdf_cdf.html) | Interactive: cutting-drill speed by receiver career tier |
| [`speed_pdf_cdf_interactive.html`](./speed_pdf_cdf_interactive.html) | Interactive: cutting-drill speed for the four case-study receivers |
| [`route_playback_interactive.html`](./route_playback_interactive.html) | Interactive: 10 Hz comeback-route playback |
| [`results/`](./results) | Model tables, summaries and the three pre-registrations printed or written by the scripts |

---

## Methodology

* **Metrics.**
  * WR median speed in the four cutting drills ($s_{\text{cut}}$).
  * WR drill-standardised cut speed ($z_{\text{cut}}$): per-drill median, z-scored within the drill, then averaged. This matters because prospects ran different drill sets.
  * DL and OL Gamma-Over-Expected: 90th-percentile acceleration magnitude residualised on weight. For DL this is computed both over all drills and within the pass-rush drill.
  * DB 90th-percentile speed in the 45° transition drill.
* **Outcomes.**
  * Career snaps.
  * Starter status (10+ starts).
  * Regular-season per-snap efficiency from `player_play.csv` (preseason and playoff snaps excluded): yards per route run, targets per route, separation at the throw, pressure rate, get-off time.
* **Models.** OLS with HC1 robust standard errors, or logit for starter status. Each is fit raw and with $\log(\text{pick})$, a UDFA flag, draft-class fixed effects and (DL/DB/OL) position fixed effects. Holm correction is applied across the 22 controlled tests.
* **Note on `a`.** The tracking column `a` is a magnitude, so acceleration percentiles include braking. The cut analyses rebuild the acceleration vector from smoothed positions (Savitzky–Golay) and split it into braking and turning force.
* **Cut analyses.** A cut is the slowest frame of a turn of 45° or more. Combine and game cuts are measured identically. The within-player test regresses separation at the throw on the last cut's speed (z-scored within route and turn angle), with player, route, coverage, alignment, down and opponent fixed effects and player-clustered errors. Conformal checks use CV+ intervals with permutation tests.

---

## Repository Structure

```text
nfl-big-data-bowl-2027/
├── REPORT.md / REPORT.html / REPORT.pdf   # Competition write-up
├── report_assigned.tex / .pdf             # Formal 4-page report
├── presentation.tex / .pdf                # Beamer deck
├── results/                               # Outputs of the analysis scripts
├── figures/                               # Copies of the static figures
├── scripts/
│   ├── common.py                          # Paths, loaders, shared definitions
│   ├── extract_features.py                # Combine features + in-game efficiency
│   ├── run_regressions.py                 # Every model, test and summary number
│   ├── rank_predictors.py                 # Predictor ranking (playing time, efficiency)
│   ├── cod_features.py                    # Combine cut speed and force metrics
│   ├── cod_force_models.py                # Combine cut metrics -> careers (+ conformal)
│   ├── game_cod_features.py               # In-game cuts from game tracking
│   ├── transfer_test.py                   # Combine vs in-game cut metrics
│   ├── play_level_fe.py                   # Within-player play-level test
│   ├── distribution_checks.py             # Wild bootstrap, permutations, distribution regression, PDF/CDF figures
│   ├── last_cut_speed.py                  # PDF/CDF of speed at the last cut (Stage 3 sample)
│   ├── followup_features.py               # One tracking pass: catch speed, get-offs, shoulders, paths, peak speed
│   ├── followup_tests.py                  # The seven pre-registered follow-up tests (T1-T7)
│   ├── followup_posthoc.py                # Post-hoc checks of T2, T3, T4 and T6 (labelled, not pre-registered)
│   ├── game_speed_distribution.py         # Regular-season speed distribution
│   ├── make_field_plot.py                 # Static figures
│   └── make_interactive_playback.py       # Plotly widgets
└── requirements.txt
```

---

## Reproducing

```bash
pip install -r requirements.txt
kaggle competitions download -c nfl-big-data-bowl-2027 -p ../data/
unzip ../data/nfl-big-data-bowl-2027.zip -d ../data/
# CSVs are read from ../data/nfl-big-data-bowl-2027 (override with NFL_BDB_DATA)

python scripts/extract_features.py
python scripts/run_regressions.py
python scripts/rank_predictors.py
python scripts/cod_features.py
python scripts/cod_force_models.py
python scripts/game_cod_features.py            # or set NFL_BDB_ZIP to read tracking from the zip
python scripts/transfer_test.py
python scripts/play_level_fe.py
python scripts/distribution_checks.py          # slow: ~30 min (conformal shuffles)
python scripts/last_cut_speed.py
python scripts/followup_features.py            # streams game tracking (~5 min)
python scripts/followup_tests.py
python scripts/followup_posthoc.py
python scripts/game_speed_distribution.py
python scripts/make_field_plot.py
python scripts/make_interactive_playback.py
```

---

## Citation
```bibtex
@article{polley2027behinddraftboard,
  title={Beyond the Stopwatch, Behind the Draft Board: What Combine Tracking Adds Once Draft Position Is Known},
  author={Polley, Michael},
  journal={NFL Big Data Bowl 2027},
  year={2026}
}
```
