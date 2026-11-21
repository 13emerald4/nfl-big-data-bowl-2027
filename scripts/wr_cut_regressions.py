"""
wr_cut_regressions.py
Baseline regressions of receivers' Combine cut speed (s_cut, km/h; and the
drill-standardised s_cut_z) on the 40 time, draft slot and body weight, in the
style of mass_vs_gamma_scatter.png.

  1. s_cut ~ forty            wr_forty_vs_cut_scatter.png
  2. s_cut ~ log_pick         wr_pick_vs_cut_scatter.png   (UDFA coded as pick 262)
  3. s_cut ~ combine_weight   wr_weight_vs_cut_scatter.png
  plus wr_cut_regressions_3panel.png (repo root and figures/).

All standard errors are HC1. Residual checks relate each "cut speed over
expected" residual to starter status (linear probability), career offensive
snaps and yards per route run (receivers with >= 100 routes), HC1, no other
controls. Writes results/wr_cut_regressions.csv and .txt.

Input: data/features_wr.parquet (extract_features.py).
"""
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.text import Text
from matplotlib.transforms import Bbox

import common as cm
from make_field_plot import plt, save, BLUE, ORANGE, INK, INK2, SURFACE

MIN_ROUTES = 100
CASE_IDS = {56042: "Puka Nacua", 55884: "Jaxon Smith-Njigba", 57144: "Brian Thomas Jr.", 55885: "Quentin Johnston"}
PICK_TICKS = [(1, "1"), (10, "10"), (32, "32"), (64, "64"), (128, "128"), (262, "UDFA")]
SPECS = [
    dict(x="forty", file="wr_forty_vs_cut_scatter.png", xlabel="Official 40-yard dash (s)",
         short="40 time", title="the 40 time barely predicts cut speed", expect="given 40 time"),
    dict(x="log_pick", file="wr_pick_vs_cut_scatter.png", xlabel="Overall draft pick (log scale; UDFA = 262)",
         short="draft pick", title="earlier picks cut faster", expect="given draft slot"),
    dict(x="combine_weight", file="wr_weight_vs_cut_scatter.png", xlabel="Combine weight (lb)",
         short="weight", title="body weight does not predict cut speed", expect="given weight"),
]
OUTCOMES = [("starter", "10+ career starts (linear probability)"), ("career_offensive_snaps", "career offensive snaps"),
            ("yprr", f"yards per route run (>= {MIN_ROUTES} routes)")]


def ols(y, x, d):
    m = smf.ols(f"{y} ~ {x}", d).fit(cov_type="HC1")
    lo, hi = m.conf_int().loc[x]
    return m, dict(coef=m.params[x], se=m.bse[x], t=m.tvalues[x], p=m.pvalues[x], ci_lo=lo, ci_hi=hi,
                   r2=m.rsquared, n=int(m.nobs))


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def scatter(ax, d, spec, m, stats, full_labels=True):
    x = spec["x"]
    for flag, color, lab in [(0, BLUE, "Fewer than 10 starts"), (1, ORANGE, "10+ starts")]:
        s = d[d.starter == flag]
        ax.scatter(s[x], s.s_cut, s=42, color=color, edgecolor=SURFACE, linewidth=1.2, label=lab, zorder=3)
    pad = 0.04 * (d[x].max() - d[x].min())
    g = np.linspace(d[x].min() - pad, d[x].max() + pad, 100)
    ax.plot(g, m.params.Intercept + m.params[x] * g, color=INK2, lw=2, ls="--",
            label=f"Expected {spec['expect']} (HC1 slope {stats['coef']:+.3f}, R² {stats['r2']:.2f}, {fmt_p(stats['p'])})")

    lab = d[d.nfl_id.isin(CASE_IDS)].copy()
    lab["name"] = lab.nfl_id.map(CASE_IDS)
    top = d.loc[d.resid.idxmax()]
    if top.nfl_id not in CASE_IDS:
        lab = pd.concat([lab, top.to_frame().T.assign(name=top.display_name)])
    span_y = d.s_cut.max() - d.s_cut.min()
    ax.set_ylim(d.s_cut.min() - 0.15 * span_y, d.s_cut.max() + 0.45 * span_y)
    if x == "log_pick":
        ax.set_xlim(np.log(0.8), np.log(262) + 0.25)
        ax.set_xticks([np.log(p) for p, _ in PICK_TICKS])
        ax.set_xticklabels([t for _, t in PICK_TICKS])
    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel("Median speed in cutting drills, s_cut (km/h)")
    lab = pd.concat([lab[lab.nfl_id == top.nfl_id], lab[lab.nfl_id != top.nfl_id]])
    place_labels(ax, d[x].to_numpy(float), d.s_cut.to_numpy(float), lab, x, top.nfl_id, full_labels)


def place_labels(ax, px, py, lab, x, top_id, full_labels):
    """Put each label next to its point at the first candidate offset whose text box overlaps no other
    label, no data point and not the legend corner (lower left); fall back to the least-crowded one."""
    fig = ax.figure
    rend = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.column_stack([px, py]))
    axbox = ax.get_window_extent(rend)
    legend = (axbox.x0, axbox.y0, axbox.x0 + 0.48 * axbox.width, axbox.y0 + 0.24 * axbox.height)
    placed, leaders = [], []
    dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1), (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 0.5), (-1, 0.5), (1, -0.5), (-1, -0.5), (0.5, 1), (-0.5, 1), (0.5, -1), (-0.5, -1)]
    cands = [(dx * r, dy * r) for r in (14, 26, 40, 60, 85, 115, 150) for dx, dy in dirs]
    for _, r in lab.iterrows():
        text = f"{r['name']} ({r.resid:+.2f} over expected)" if full_labels else f"{r['name']} ({r.resid:+.2f})"
        if r.nfl_id == top_id:
            text += ", top residual"
        best, best_cost = None, None
        for dx, dy in cands:
            ann = ax.annotate(text, (float(r[x]), float(r.s_cut)), xytext=(dx, dy), textcoords="offset points",
                              ha="left" if dx > 0 else "right" if dx < 0 else "center",
                              va="bottom" if dy > 0 else "top" if dy < 0 else "center",
                              fontsize=8.5 if full_labels else 7.8, color=INK, zorder=4,
                              arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkA=1, shrinkB=3))
            ann.update_positions(rend)
            raw = Text.get_window_extent(ann, rend)                       # text only, not the leader
            b = raw.expanded(1.04, 1.25)
            inside = b.x0 >= axbox.x0 and b.x1 <= axbox.x1 and b.y0 >= axbox.y0 and b.y1 <= axbox.y1
            cost = 0 if inside else 1000
            cost += 1000 * sum(b.overlaps(p) for p in placed)
            cost += 40 * np.sum((pts[:, 0] > b.x0 - 4) & (pts[:, 0] < b.x1 + 4) & (pts[:, 1] > b.y0 - 4) & (pts[:, 1] < b.y1 + 4))
            cost += 1000 * (b.x0 < legend[2] and b.y0 < legend[3])
            # leader: from the point to the nearest spot on the text box; the drawn line is pinned there (relpos)
            p0 = pts_of(ax, float(r[x]), float(r.s_cut))
            p1 = (min(max(p0[0], raw.x0), raw.x1), min(max(p0[1], raw.y0), raw.y1))
            ann._arrow_relpos = (   # read at draw time when an arrowstyle is set (matplotlib 3.7)
                (p1[0] - raw.x0) / max(raw.width, 1), (p1[1] - raw.y0) / max(raw.height, 1))
            seg = (p0, p1)
            cost += 500 * sum(crosses(seg, q) for q in placed)              # my leader through another label
            cost += 500 * sum(crosses(s, b) for s in leaders)               # another leader through my label
            cost += 300 * sum(intersect(seg, s) for s in leaders)           # leaders crossing each other
            cost += 15 * near_points(seg, pts)                              # leader running over data points
            clash = cost
            cost += 0.004 * (dx * dx + dy * dy)                             # prefer short leaders
            if best_cost is None or cost < best_cost:
                if best is not None:
                    best[0].remove()
                best, best_cost = (ann, b, seg), cost
            else:
                ann.remove()
            if clash == 0:      # candidates run shortest first: the first clean spot wins
                break
        placed.append(Bbox(best[1].get_points().copy()))
        leaders.append(best[2])


def intersect(a, b):
    """Proper intersection of two segments."""
    (p1, p2), (p3, p4) = a, b

    def orient(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return bool(orient(p1, p2, p3) * orient(p1, p2, p4) < 0 and orient(p3, p4, p1) * orient(p3, p4, p2) < 0)


def near_points(seg, pts, tol=5.0):
    """Data points within tol pixels of the leader, away from its ends."""
    (x0, y0), (x1, y1) = seg
    v = np.array([x1 - x0, y1 - y0])
    L2 = float(v @ v)
    if L2 < 1:
        return 0
    w = pts - np.array([x0, y0])
    s = np.clip((w @ v) / L2, 0, 1)
    dist = np.hypot(*(w - np.outer(s, v)).T)
    return int(np.sum((dist < tol) & (s > 0.1) & (s < 0.9)))


def pts_of(ax, x, y):
    return tuple(ax.transData.transform((x, y)))


def crosses(seg, box):
    """Does the segment pass through the box (sampled; endpoints excluded)?"""
    (x0, y0), (x1, y1) = seg
    t = np.linspace(0.08, 0.92, 25)
    xs, ys = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    return bool(np.any((xs > box.x0) & (xs < box.x1) & (ys > box.y0) & (ys < box.y1)))


def main():
    wr = pd.read_parquet(os.path.join(cm.OUT_DIR, "features_wr.parquet"))
    rows, fits = [], {}
    for spec in SPECS:
        x = spec["x"]
        d = wr.dropna(subset=[x, "s_cut"]).copy()
        for y in ["s_cut", "s_cut_z"]:
            m, st = ols(y, x, d)
            rows.append(dict(section="baseline", regression=f"s_cut ~ {x}", outcome=y, term=x, **st))
            if y == "s_cut":
                d["resid"] = m.resid
                fits[x] = (d, m, st)
            else:
                d["resid_z"] = m.resid

        # residual -> outcome checks
        for rcol, rname in [("resid", "s_cut"), ("resid_z", "s_cut_z")]:
            for out, _ in OUTCOMES:
                s = d[d.routes >= MIN_ROUTES] if out == "yprr" else d
                s = s.dropna(subset=[out])
                _, st = ols(out, rcol, s)
                rows.append(dict(section="residual check", regression=f"{rname} over expected | {x}",
                                 outcome=out, term=f"{rname} residual", **st))

        # top / bottom 5 over expected (s_cut)
        rank = d.sort_values("resid", ascending=False)
        for tag, part in [("top 5", rank.head(5)), ("bottom 5", rank.tail(5).iloc[::-1])]:
            for i, r in enumerate(part.itertuples(), 1):
                rows.append(dict(section=tag, regression=f"s_cut ~ {x}", outcome="s_cut", term="residual",
                                 rank=i, player=r.display_name, value_x=getattr(r, x), s_cut=r.s_cut,
                                 resid=r.resid, starter=r.starter, career_offensive_snaps=r.career_offensive_snaps,
                                 draft_pick=r.draft_overall_pick if r.udfa == 0 else np.nan))

        fig, ax = plt.subplots(figsize=(8.5, 5))
        dd, m, st = fits[x]
        scatter(ax, dd, spec, m, st)
        ax.set_title(f"Wide receivers (N={len(dd)}): {spec['title']} (labels: km/h over expected)", loc="left",
                     fontsize=11)
        ax.legend(fontsize=8.5, loc="lower left")
        save(fig, spec["file"])

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4), gridspec_kw={"wspace": 0.22})
    for ax, spec in zip(axes, SPECS):
        dd, m, st = fits[spec["x"]]
        scatter(ax, dd, spec, m, st, full_labels=False)
        ax.set_title(f"{spec['short'].capitalize()} (N={len(dd)}): slope {st['coef']:+.3f}, R² {st['r2']:.2f}, "
                     f"{fmt_p(st['p'])}", loc="left", fontsize=10.5)
        if ax is not axes[0]:
            ax.set_ylabel("")
        h, _ = ax.get_legend_handles_labels()
        if ax is axes[0]:
            ax.legend(h, ["Fewer than 10 starts", "10+ starts", "HC1 OLS fit"], fontsize=8.3, loc="lower left")
        else:
            ax.legend(h[2:], ["HC1 OLS fit"], fontsize=8.3, loc="lower left")
    fig.suptitle("Receivers' Combine cut speed against the 40 time, draft slot and body weight "
                 "(labels: km/h over expected)", x=0.07, ha="left", y=1.0, fontsize=11.5, color=INK)
    save(fig, "wr_cut_regressions_3panel.png")

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(cm.RESULTS_DIR, "wr_cut_regressions.csv"), index=False)
    lines = ["Receivers' Combine cut speed: baseline regressions (HC1 standard errors)",
             f"Input data/features_wr.parquet, {len(wr)} receivers. s_cut = median speed in cutting drills (km/h); "
             "s_cut_z = drill-standardised.", ""]
    for x in [s["x"] for s in SPECS]:
        lines.append(f"== s_cut ~ {x}")
        for r in R[(R.section == "baseline") & (R.term == x)].itertuples():
            lines.append(f"  {r.outcome:8s} slope {r.coef:+.4f} (SE {r.se:.4f}) [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}], "
                         f"{fmt_p(r.p)}, R² {r.r2:.3f}, N = {int(r.n)}")
        lines.append("  Residual checks (outcome ~ cut speed over expected, HC1):")
        for r in R[(R.section == "residual check") & (R.regression.str.endswith(f"| {x}"))].itertuples():
            lines.append(f"    {r.regression.split(' |')[0]:28s} -> {r.outcome:24s} {r.coef:+.4f} "
                         f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}], {fmt_p(r.p)}, N = {int(r.n)}")
        for tag in ["top 5", "bottom 5"]:
            part = R[(R.section == tag) & (R.regression == f"s_cut ~ {x}")]
            lines.append(f"  {tag.capitalize()} over expected: " + "; ".join(
                f"{r.player} {r.resid:+.2f}" for r in part.itertuples()))
        lines.append("")
    miss = wr[wr.forty.isna() & wr.nfl_id.isin(CASE_IDS)].nfl_id.map(CASE_IDS).tolist()
    if miss:
        lines.append(f"No official 40 time (absent from regression 1): {', '.join(miss)}.")
    txt = "\n".join(lines)
    with open(os.path.join(cm.RESULTS_DIR, "wr_cut_regressions.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
