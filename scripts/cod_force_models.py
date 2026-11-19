"""
cod_force_models.py
Runs the plan in results/cod_force_preregistration.md on the metrics from
cod_features.py.

Frequentist part. Each metric -> outcome model is fit twice: alone (total
association; position fixed effects in multi-position samples) and with draft
controls. Estimators follow the outcome:
  YPRR, separation                 OLS, HC1 standard errors
  targets per route, pressure rate fractional logit (Papke-Wooldridge), HC1
  snaps per season                 Poisson QMLE, offset log(seasons), HC1
  starter (>= 10 starts)           logit, HC1
Effects are per 1 SD of the metric, with 95% confidence intervals, and as
average partial effects (APE) in outcome units. The APE interval scales the
coefficient interval by the sample-average derivative (it ignores the
derivative's own sampling noise). Holm correction: the two primary tests as
one family, every other controlled test as a second family.

Conformal part. CV+ prediction intervals (Barber, Candes, Ramdas & Tibshirani
2021) at 80% for each player, from the controlled model with and without the
metric. Each player's interval is built without that player (10 outer folds,
5 inner folds, fixed splits shared by every model). Reported: coverage, mean
width, width change; the permutation p-value is the share of 1,000 shuffles of
the metric that narrow the intervals at least as much.

Writes results/cod_force_models.csv, results/cod_force_conformal.csv and
results/cod_force_summary.txt.
"""
import os

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

import common as cm

CONTROLS = "log_pick + udfa + C(draft_year)"
MIN_ROUTES, MIN_TARGETS, MIN_RUSHES, MIN_N = 100, 20, 100, 25
ALPHA = 0.20
N_PERM = 1000
SEED = 2027

IDEA1_WR = ["cod_speed_z", "cod_retention_z", "cod_exit_z"]
IDEA2 = ["force_oe_z", "force_per_kg_z", "force_abs_z", "brake_oe_z", "turn_oe_z", "fv_f0", "fv_slope"]
NEEDS_MASS = {"force_abs_z", "force_per_kg_z", "fv_f0", "fv_slope"}
PRIMARY = {("cod_speed_z", "yprr", "WR"), ("force_oe_z", "snaps_per_season", "ALL")}


# ---------------------------------------------------------------- data
def load():
    base = cm.player_base()
    cod = pd.read_parquet(os.path.join(cm.OUT_DIR, "features_cod.parquet"))
    cod = cod.drop(columns=["display_name", "nfl_position", "draft_year", "combine_weight"])
    df = base.merge(cod, on="nfl_id")
    df["mass_kg"] = df.combine_weight * cm.LB_TO_KG
    df["snaps"] = df.career_offensive_snaps.fillna(0) + df.career_defensive_snaps.fillna(0)
    df["snaps_per_season"] = df.snaps / df.seasons
    df["log_seasons"] = np.log(df.seasons)
    for c in ["fv_f0", "fv_slope"]:                    # put on a per-SD scale like the others
        df[c] = (df[c] - df[c].mean()) / df[c].std()
    wr_eff = pd.read_parquet(os.path.join(cm.OUT_DIR, "features_wr.parquet"))[
        ["nfl_id", "routes", "yprr", "targets_per_route", "separation", "n_separation"]]
    dl_eff = pd.read_parquet(os.path.join(cm.OUT_DIR, "features_dl.parquet"))[["nfl_id", "rush_snaps", "pressure_rate"]]
    df = df.merge(wr_eff, on="nfl_id", how="left").merge(dl_eff, on="nfl_id", how="left")
    df["edge"] = df.nfl_position.isin(["DE", "OLB"]).astype(int)
    return df


def samples(df):
    wr = df[df.pos_group == "WR"]
    allp = df[df.pos_group != "QB"]
    dl = df[df.pos_group == "DL"]
    return {
        ("WR", "yprr"): wr[wr.routes >= MIN_ROUTES],
        ("WR", "targets_per_route"): wr[wr.routes >= MIN_ROUTES],
        ("WR", "separation"): wr[wr.n_separation >= MIN_TARGETS],
        ("WR", "snaps_per_season"): wr,
        ("WR", "starter"): wr,
        ("ALL", "snaps_per_season"): allp,
        ("ALL", "starter"): allp,
        ("DL", "pressure_rate"): dl[dl.rush_snaps >= MIN_RUSHES],
    }


KIND = {"yprr": "ols", "separation": "ols", "targets_per_route": "frac", "pressure_rate": "frac",
        "snaps_per_season": "poisson", "starter": "logit"}
UNITS = {"yprr": "yards per route", "separation": "yards", "targets_per_route": "targets per route",
         "pressure_rate": "pressures per rush", "snaps_per_season": "snaps per season",
         "starter": "probability of starting"}


# ---------------------------------------------------------------- frequentist
def fit(kind, y, rhs, d):
    if kind == "poisson":
        return smf.glm(f"snaps ~ {rhs}", d, family=sm.families.Poisson(),
                       offset=d.log_seasons).fit(cov_type="HC1", maxiter=200)
    if kind in ("frac", "logit"):
        return smf.glm(f"{y} ~ {rhs}", d, family=sm.families.Binomial()).fit(cov_type="HC1", maxiter=200)
    return smf.ols(f"{y} ~ {rhs}", d).fit(cov_type="HC1")


def ape_factor(kind, m, d):
    """Sample-average derivative of E[y|x] with respect to the linear index."""
    if kind == "ols":
        return 1.0
    mu = m.fittedvalues if kind != "poisson" else m.fittedvalues / d.seasons
    return float(np.mean(mu * (1 - mu))) if kind in ("frac", "logit") else float(np.mean(mu))


def rhs_for(metric, group, controlled, extra):
    terms = [metric]
    if group == "ALL":
        terms.append("C(pos_group)")
    if group == "DL":
        terms.append("edge")
    if metric in NEEDS_MASS:
        terms.append("mass_kg")
    if controlled:
        terms.append(CONTROLS)
    return " + ".join(terms + extra)


def frequentist(df):
    rows = []
    S = samples(df)
    plan = [(m, "WR", y) for m in IDEA1_WR for y in ["yprr", "targets_per_route", "separation", "snaps_per_season", "starter"]]
    plan += [("tc_cut_speed_z", "ALL", y) for y in ["snaps_per_season", "starter"]]
    plan += [(m, "ALL", y) for m in IDEA2 for y in ["snaps_per_season", "starter"]]
    plan += [(m, "DL", "pressure_rate") for m in IDEA2]
    for metric, group, y in plan:
        d = S[(group, y)]
        need = [metric, y, "log_pick", "udfa", "draft_year"] + (["mass_kg"] if metric in NEEDS_MASS else [])
        d = d.dropna(subset=need)
        if len(d) < MIN_N:
            continue
        kind = KIND[y]
        for controlled in (False, True):
            try:
                m = fit(kind, y, rhs_for(metric, group, controlled, []), d)
            except Exception as e:                      # perfect separation etc.
                print(f"skip {metric} -> {y} ({group}, controlled={controlled}): {e}")
                continue
            b, se, p = m.params[metric], m.bse[metric], m.pvalues[metric]
            lo, hi = m.conf_int().loc[metric]
            f = ape_factor(kind, m, d)
            rows.append(dict(
                idea=1 if metric in IDEA1_WR or metric == "tc_cut_speed_z" else 2,
                metric=metric, group=group, outcome=y, estimator=kind, controlled=controlled, n=len(d),
                coef=b, se=se, ci_lo=lo, ci_hi=hi, p=p,
                ape=b * f, ape_lo=lo * f, ape_hi=hi * f, ape_units=UNITS[y],
                pct_effect=(np.exp(b) - 1) * 100 if kind == "poisson" else np.nan,
                primary=(metric, y, group) in PRIMARY and controlled))
    R = pd.DataFrame(rows)
    R["p_holm"] = np.nan
    for mask in [R.primary, R.controlled & ~R.primary]:
        if mask.sum():
            R.loc[mask, "p_holm"] = multipletests(R.loc[mask, "p"], method="holm")[1]
    return R


# ---------------------------------------------------------------- conformal
def design(formula, d):
    y, X = patsy.dmatrices(formula, d, return_type="dataframe")
    return X.to_numpy(float)


def fit_predict(kind, X, y, off, Xt, offt):
    """Point prediction on the outcome's rate scale (snaps per season for Poisson)."""
    if kind == "ols":
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return Xt @ b
    # Poisson IRLS on counts with offset, small ridge for stability with sparse dummies.
    b = np.zeros(X.shape[1])
    b[0] = np.log(max(np.mean(y / np.exp(off)), 1e-3))
    ridge = 1e-6 * np.eye(X.shape[1])
    for _ in range(50):
        eta = np.clip(X @ b + off, -30, 30)
        mu = np.exp(eta)
        z = eta - off + (y - mu) / mu
        W = mu
        b_new = np.linalg.solve(X.T @ (W[:, None] * X) + ridge, X.T @ (W * z))
        if np.max(np.abs(b_new - b)) < 1e-8:
            b = b_new
            break
        b = b_new
    return np.exp(np.clip(Xt @ b, -30, 30))


def cv_plus(kind, X, y, off, target, outer, inner_k=5, alpha=ALPHA, rng=None):
    """CV+ intervals for each point, built from the other folds only."""
    n = len(y)
    lo, hi = np.empty(n), np.empty(n)
    for test in outer:
        train = np.setdiff1d(np.arange(n), test)
        folds = np.array_split(rng.permutation(train), inner_k)
        preds_test, resid = [], []
        for f in folds:
            tr = np.setdiff1d(train, f)
            p_all = fit_predict(kind, X[tr], y[tr], off[tr], X[np.r_[f, test]], off[np.r_[f, test]])
            p_f, p_t = p_all[:len(f)], p_all[len(f):]
            r = np.abs(target[f] - p_f)
            resid.append(r)
            preds_test.append(np.repeat(p_t[None, :], len(f), 0))   # rows: training points in fold f
        R = np.concatenate(resid)[:, None]
        P = np.concatenate(preds_test, 0)
        m = len(R)
        k_hi = int(np.ceil((1 - alpha) * (m + 1))) - 1
        k_lo = int(np.floor(alpha * (m + 1))) - 1
        lo[test] = np.sort(P - R, 0)[max(k_lo, 0)]
        hi[test] = np.sort(P + R, 0)[min(k_hi, m - 1)]
    return lo, hi


def conformal_one(df_s, metric, y, group):
    kind = "poisson" if KIND[y] == "poisson" else "ols"
    need = [metric, y, "log_pick", "udfa", "draft_year"] + (["mass_kg"] if metric in NEEDS_MASS else [])
    d = df_s.dropna(subset=need).reset_index(drop=True)
    resp = "snaps" if kind == "poisson" else y
    f_base = f"{resp} ~ {rhs_for(metric, group, True, [])}".replace(f"{metric} + ", "")
    f_full = f"{resp} ~ {rhs_for(metric, group, True, [])}"
    Xb, Xf = design(f_base, d), design(f_full, d)
    j = list(patsy.dmatrices(f_full, d, return_type="dataframe")[1].columns).index(metric)
    yv = d[resp].to_numpy(float)
    off = d.log_seasons.to_numpy(float) if kind == "poisson" else np.zeros(len(d))
    target = d[y].to_numpy(float)
    rng = np.random.default_rng(SEED)
    outer = np.array_split(rng.permutation(len(d)), 10)
    state = rng.bit_generator.state

    def run(X):
        rng.bit_generator.state = state                  # identical inner splits for every model
        lo, hi = cv_plus(kind, X, yv, off, target, outer, rng=rng)
        return lo, hi

    lb, hb = run(Xb)
    lf, hf = run(Xf)
    wb, wf = np.mean(hb - lb), np.mean(hf - lf)
    cov_b = np.mean((target >= lb) & (target <= hb))
    cov_f = np.mean((target >= lf) & (target <= hf))
    obs = wb - wf
    prng = np.random.default_rng(SEED + 1)
    null = np.empty(N_PERM)
    for b in range(N_PERM):
        Xp = Xf.copy()
        Xp[:, j] = prng.permutation(Xp[:, j])
        lp, hp = run(Xp)
        null[b] = wb - np.mean(hp - lp)
    p = (1 + np.sum(null >= obs)) / (N_PERM + 1)
    return dict(metric=metric, group=group, outcome=y, n=len(d), coverage_target=1 - ALPHA,
                coverage_base=cov_b, coverage_with_metric=cov_f,
                width_base=wb, width_with_metric=wf, width_change_pct=(wf - wb) / wb * 100,
                null_median_change_pct=-np.median(null) / wb * 100, perm_p=p,
                primary=(metric, y, group) in PRIMARY)


def conformal(df, R):
    S = samples(df)
    todo = sorted({(r.metric, r.outcome, r.group) for r in R.itertuples()
                   if r.controlled and KIND[r.outcome] in ("ols", "poisson")},
                  key=lambda t: ((t[0], t[1], t[2]) not in PRIMARY, t))
    out = []
    for metric, y, group in todo:
        out.append(conformal_one(S[(group, y)], metric, y, group))
        o = out[-1]
        print(f"conformal {metric:16s} -> {y:18s} ({group}): width {o['width_base']:.3f} -> "
              f"{o['width_with_metric']:.3f} ({o['width_change_pct']:+.1f}%), perm p = {o['perm_p']:.3f}")
    return pd.DataFrame(out)


# ---------------------------------------------------------------- report
def summary(R, C, df):
    L = ["Change-of-direction speed and force -> NFL outcomes",
         "Plan: results/cod_force_preregistration.md; metrics: results/cod_reliability.txt", ""]
    L.append("PRIMARY TESTS (draft-controlled; Holm across the two)")
    for r in R[R.primary].itertuples():
        raw = R[(R.metric == r.metric) & (R.outcome == r.outcome) & (R.group == r.group) & ~R.controlled].iloc[0]
        L.append(f"  {r.metric} -> {r.outcome} ({r.group}, N = {r.n}, {r.estimator})")
        L.append(f"    controlled: coef {r.coef:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}], p = {r.p:.3f}, Holm p = {r.p_holm:.3f}")
        L.append(f"      per 1 SD: {r.ape:+.3f} {r.ape_units} [{r.ape_lo:+.3f}, {r.ape_hi:+.3f}]"
                 + (f"; {r.pct_effect:+.1f}% snaps" if r.estimator == "poisson" else ""))
        L.append(f"    alone:      coef {raw.coef:+.3f} [{raw.ci_lo:+.3f}, {raw.ci_hi:+.3f}], p = {raw.p:.3f}")
        c = C[(C.metric == r.metric) & (C.outcome == r.outcome) & (C.group == r.group)]
        if len(c):
            c = c.iloc[0]
            L.append(f"    conformal 80%: coverage {c.coverage_base:.2f} -> {c.coverage_with_metric:.2f}; "
                     f"mean width {c.width_base:.3f} -> {c.width_with_metric:.3f} ({c.width_change_pct:+.1f}%); "
                     f"noise metric median {c.null_median_change_pct:+.1f}%; permutation p = {c.perm_p:.3f}")
    L.append("")
    ex = R[R.controlled & ~R.primary].sort_values("p")
    L.append(f"EXPLORATORY ({len(ex)} controlled tests; Holm within the family)")
    L.append(f"  p < 0.05: {(ex.p < 0.05).sum()} (expected by chance: {0.05 * len(ex):.1f}); "
             f"Holm p < 0.05: {(ex.p_holm < 0.05).sum()}")
    for r in ex.head(8).itertuples():
        L.append(f"  {r.metric:16s} -> {r.outcome:18s} {r.group:3s} N={r.n:3d}  per SD {r.ape:+.3f} "
                 f"[{r.ape_lo:+.3f}, {r.ape_hi:+.3f}] {r.ape_units}; p = {r.p:.3f}, Holm {r.p_holm:.2f}")
    L.append("")
    L.append("CONFORMAL (80% CV+ intervals; negative change = metric narrows intervals)")
    for c in C.itertuples():
        L.append(f"  {c.metric:16s} -> {c.outcome:18s} {c.group:3s} N={c.n:3d}  coverage {c.coverage_with_metric:.2f}  "
                 f"width {c.width_change_pct:+.1f}% (noise {c.null_median_change_pct:+.1f}%)  perm p = {c.perm_p:.3f}")
    L.append(f"  permutation p < 0.05: {(C.perm_p < 0.05).sum()} of {len(C)}")
    return "\n".join(L)


def main():
    df = load()
    R = frequentist(df)
    C = conformal(df, R)
    R.to_csv(os.path.join(cm.RESULTS_DIR, "cod_force_models.csv"), index=False)
    C.to_csv(os.path.join(cm.RESULTS_DIR, "cod_force_conformal.csv"), index=False)
    txt = summary(R, C, df)
    with open(os.path.join(cm.RESULTS_DIR, "cod_force_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
