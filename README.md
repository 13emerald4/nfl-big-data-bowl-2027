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
* Separation at the throw does **not** rise with cut speed.
* **Predictor ranking** (`scripts/rank_predictors.py`):
  * **Playing time:** draft pick ranks first at every position (starter AUC 0.82–0.93), then the NGS final and college-production scores. Nothing adds to draft pick out of sample.
  * **Per-snap efficiency:** the strongest signals beyond draft slot are drill-standardised cut speed for receiver YPRR (+0.33, p = 0.008), and the NGS final score (+0.36) and 10-yard split (−0.29, p = 0.034) for pass-rush pressure rate.
  * Exploratory: none of the 30 efficiency correlations survives Holm correction.

---

## Artifacts

| Artifact | Description |
| :--- | :--- |
| [`REPORT.md`](./REPORT.md) / [`REPORT.html`](./REPORT.html) / [`REPORT.pdf`](./REPORT.pdf) | Competition write-up (< 2,000 words, 4 figures, 5 tables) |
| [`report_assigned.pdf`](./report_assigned.pdf) | 4-page formal report (LaTeX) |
| [`presentation.pdf`](./presentation.pdf) | 10-slide Beamer deck |
| [`draft_controls.png`](./draft_controls.png) | Test statistic for every model, before vs after draft controls |
| [`wr_speed_pdf_cdf.png`](./wr_speed_pdf_cdf.png) | Cutting-drill speed density and CDF for four case-study receivers |
| [`route_kinematics_field_plot.png`](./route_kinematics_field_plot.png) | Comeback route: Nacua vs Johnston paths and speed through the break |
| [`mass_vs_gamma_scatter.png`](./mass_vs_gamma_scatter.png) | Defensive-line weight vs 90th-percentile acceleration |
| [`tier_comparison_pdf_cdf.html`](./tier_comparison_pdf_cdf.html) | Interactive: cutting-drill speed by receiver career tier |
| [`speed_pdf_cdf_interactive.html`](./speed_pdf_cdf_interactive.html) | Interactive: cutting-drill speed for the four case-study receivers |
| [`route_playback_interactive.html`](./route_playback_interactive.html) | Interactive: 10 Hz comeback-route playback |
| [`results/`](./results) | Model table (`model_results.csv`), descriptives and game-speed summary printed by the scripts |

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
* **Note on `a`.** The tracking column `a` is a magnitude, so acceleration percentiles include braking.

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
python scripts/game_speed_distribution.py      # or set NFL_BDB_ZIP to read tracking from the zip
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
