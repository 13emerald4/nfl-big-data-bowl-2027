"""
metric_search_ga.py
Runs results/ga_preregistration.md on data/ga_cuts.parquet (ga_cut_features.py):

  * NSGA-II over cut-detection setting, peer group, metric choice and weights,
    on the 2023-24 seasons: maximise the within-receiver effect on separation,
    keep split-half reliability >= 0.60, minimise the number of metrics;
  * a null of 200 complete searches with separation residuals permuted within
    receiver (Freedman-Lane);
  * the selected composite and the Stage 3 benchmark on the untouched 2025 season;
  * receiver rankings with bootstrap rank intervals (cut composite, momentum
    through the cut, speed for size).

Writes results/ga_summary.txt, results/ga_frontier.csv, results/athlete_rankings.csv,
ga_frontier.png and athlete_rankings.png (repo root and figures/).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import patsy
from scipy import stats

import common as cm
import cod_features as cf
import distribution_checks as DC
import play_level_fe as P
import make_field_plot as mfp
from make_field_plot import plt, BLUE, ORANGE, AQUA, INK, INK2, GRID, SURFACE

SEED = 2027
POP, GENS, N_NULL, B_RANK = 80, 40, 200, 2000
MIN_GROUP, MIN_PLAYS, MIN_HALF, REL_MIN, MAX_METRICS = 30, 30, 10, 0.60, 4
TURNS, HALVES = (30, 45, 60), (3, 5, 7)
SETTINGS = [(t, h) for t in TURNS for h in HALVES]
PEERS = ["position", "route", "route x angle"]
METRICS = ["apex", "retention", "reaccel", "recover_fast", "brake", "turnacc"]
LABEL = {"apex": "cut speed", "retention": "speed kept", "reaccel": "re-acceleration",
         "recover_fast": "recovery (faster)", "brake": "braking", "turnacc": "turning accel."}
BENCH = dict(setting=SETTINGS.index((45, 5)), peer=2, inc=[1, 0, 0, 0, 0, 0], w=[1, 0, 0, 0, 0, 0])


# ---------------------------------------------------------------- data
def build():
    base = cm.player_base()
    base["pos_group"] = base.nfl_position.map(cf.POS_GROUP)
    wr_ids = set(base.loc[base.pos_group == "WR", "nfl_id"])
    pp = cm.player_play_regular_season(usecols=P.PP_COLS)
    drop = pp.groupby(["game_id", "play_id"]).dropback_duration.max().rename("dropback").reset_index()
    games = pd.read_csv(cm.path("games.csv"))
    ga = pd.read_parquet(os.path.join(cm.OUT_DIR, "ga_cuts.parquet"))
    ga = ga[ga.nfl_id.isin(wr_ids)]
    ga["recover_fast"] = -ga.recover
    out = {}
    for si, (t, h) in enumerate(SETTINGS):
        c = ga[(ga.min_turn == t) & (ga.half == h)].drop(columns=["min_turn", "half"])
        c = c.merge(pp[["game_id", "play_id", "nfl_id", "route_ran"]], on=["game_id", "play_id", "nfl_id"])
        c = c[c.route_ran.notna()].copy()
        c["angle_bin"] = pd.cut(c.turn_deg, [t, 90, 135, 180.01], right=False).astype(str)
        c = c[c.groupby(["route_ran", "angle_bin"]).apex.transform("size") >= MIN_GROUP]
        d = c.merge(pp.drop(columns=["route_ran", "dropback_duration"]), on=["game_id", "play_id", "nfl_id"])
        d = d.merge(drop, on=["game_id", "play_id"], how="left").merge(
            games[["game_id", "season", "home_team_abbr", "visitor_team_abbr"]], on="game_id")
        d["opp_season"] = np.where(d.team_abbr == d.home_team_abbr, d.visitor_team_abbr, d.home_team_abbr) \
            + "_" + d.season.astype(str)
        d["separation"] = d.separation_at_pass_forward
        d = d.dropna(subset=["separation", "cushion", "dropback", "team_coverage_type", "receiver_alignment",
                             "down", "yards_to_go"] + METRICS)
        n = d.groupby("nfl_id").size()
        d = d[d.nfl_id.isin(n[n >= MIN_PLAYS].index)].reset_index(drop=True)
        d["position"] = "WR"
        d["route"] = d.route_ran
        d["route x angle"] = d.route_ran + "|" + d.angle_bin
        out[si] = d
        print(f"  setting {t} deg, +-{h / 10:.1f} s: {len(d):,} plays, {d.nfl_id.nunique()} receivers "
              f"(train {int((d.season <= 2024).sum()):,})", flush=True)
    return out, base


def zstats(train, peer, m):
    g = train.groupby(peer)[m].agg(["mean", "std", "size"])
    return g[g["size"] >= 10], train[m].mean(), train[m].std()


def zapply(df, peer, m, st):
    g, mu0, sd0 = st
    mu = df[peer].map(g["mean"]).fillna(mu0).to_numpy()
    sd = df[peer].map(g["std"]).fillna(sd0).to_numpy()
    return (df[m].to_numpy() - mu) / np.where(sd > 0, sd, sd0)


class Setting:
    """Everything one cut-detection setting needs for fast evaluation on the training seasons."""

    def __init__(self, d, rng):
        self.all = d
        self.train = d[d.season <= 2024].reset_index(drop=True)
        self.test = d[d.season == 2025].reset_index(drop=True)
        tr = self.train
        Z = patsy.dmatrix(f"{P.COVARS} + {P.FE}", tr, return_type="dataframe")
        assert len(Z) == len(tr)
        self.pt = DC.Partial(Z.to_numpy(float), tr.nfl_id.to_numpy())
        self.yt = self.pt.M(tr.separation.to_numpy(float))
        self.st = {(p, m): zstats(tr, p, m) for p in PEERS for m in METRICS}
        self.z = {(p, m): zapply(tr, p, m, self.st[p, m]) for p in PEERS for m in METRICS}
        self.zall = {(p, m): zapply(d, p, m, self.st[p, m]) for p in PEERS for m in METRICS}
        self.ztest = {(p, m): zapply(self.test, p, m, self.st[p, m]) for p in PEERS for m in METRICS}
        self.g = self.pt.g
        self.members = [np.flatnonzero(self.g == k) for k in range(self.pt.G)]
        half = np.empty(len(tr), int)
        for ix in self.members:
            half[ix] = rng.permutation(np.arange(len(ix)) % 2)
        self.half = half
        cnt = np.array([[np.sum(half[ix] == 0), np.sum(half[ix] == 1)] for ix in self.members])
        self.rel_ok = (cnt >= MIN_HALF).all(1)

    def composite(self, peer, w, z=None):
        z = self.z if z is None else z
        x = sum(wi * z[PEERS[peer], m] for wi, m in zip(w, METRICS) if wi != 0)
        return x

    def evaluate(self, peer, w, yt=None):
        yt = self.yt if yt is None else yt
        x = self.composite(peer, w)
        if np.isscalar(x) or x.std() < 1e-9:
            return 0.0, 0.0
        x = (x - x.mean()) / x.std()
        xt = self.pt.M(x)
        sxx = xt @ xt
        validity = float(xt @ yt / sxx) if sxx > 1e-9 else 0.0
        means = [np.bincount(self.g[self.half == h], weights=x[self.half == h], minlength=self.pt.G) /
                 np.maximum(np.bincount(self.g[self.half == h], minlength=self.pt.G), 1) for h in (0, 1)]
        r = np.corrcoef(means[0][self.rel_ok], means[1][self.rel_ok])[0, 1]
        rel = 2 * r / (1 + r) if r > -1 else -1.0
        return validity, float(rel)


# ---------------------------------------------------------------- NSGA-II
def random_genome(rng):
    inc = np.zeros(len(METRICS), int)
    inc[rng.choice(len(METRICS), rng.integers(1, MAX_METRICS + 1), replace=False)] = 1
    return dict(setting=int(rng.integers(len(SETTINGS))), peer=int(rng.integers(len(PEERS))), inc=inc,
                w=rng.uniform(-1, 1, len(METRICS)))


def repair(gn, rng):
    on = np.flatnonzero(gn["inc"])
    if len(on) == 0:
        gn["inc"][rng.integers(len(METRICS))] = 1
    while gn["inc"].sum() > MAX_METRICS:
        gn["inc"][rng.choice(np.flatnonzero(gn["inc"]))] = 0
    return gn


def weights(gn):
    return np.where(np.asarray(gn["inc"]) == 1, gn["w"], 0.0)


def objectives(gn, S, yts=None):
    s = gn["setting"]
    v, r = S[s].evaluate(gn["peer"], weights(gn), None if yts is None else yts[s])
    return np.array([-v, float(np.sum(gn["inc"]))]), max(0.0, REL_MIN - r), v, r


def dominates(a, b):
    """Constraint-domination (Deb): feasible beats infeasible; then Pareto."""
    (fa, ca), (fb, cb) = a, b
    if ca == 0 and cb > 0:
        return True
    if ca > 0 and cb > 0:
        return ca < cb
    if ca > 0 and cb == 0:
        return False
    return bool(np.all(fa <= fb) and np.any(fa < fb))


def fronts(objs):
    n = len(objs)
    S = [[] for _ in range(n)]
    cnt = np.zeros(n, int)
    F = [[]]
    for p in range(n):
        for q in range(n):
            if p != q:
                if dominates(objs[p], objs[q]):
                    S[p].append(q)
                elif dominates(objs[q], objs[p]):
                    cnt[p] += 1
        if cnt[p] == 0:
            F[0].append(p)
    i = 0
    while F[i]:
        nxt = []
        for p in F[i]:
            for q in S[p]:
                cnt[q] -= 1
                if cnt[q] == 0:
                    nxt.append(q)
        i += 1
        F.append(nxt)
    return F[:-1]


def crowding(objs, front):
    d = np.zeros(len(front))
    f = np.array([objs[i][0] for i in front])
    for j in range(f.shape[1]):
        o = np.argsort(f[:, j])
        d[o[0]] = d[o[-1]] = np.inf
        span = f[o[-1], j] - f[o[0], j]
        if span > 0:
            d[o[1:-1]] += (f[o[2:], j] - f[o[:-2], j]) / span
    return d


def vary(a, b, rng):
    c = dict(setting=a["setting"] if rng.random() < 0.5 else b["setting"],
             peer=a["peer"] if rng.random() < 0.5 else b["peer"],
             inc=np.where(rng.random(len(METRICS)) < 0.5, a["inc"], b["inc"]).astype(int))
    lam = rng.random(len(METRICS))
    c["w"] = lam * a["w"] + (1 - lam) * b["w"]
    if rng.random() < 0.1:
        c["setting"] = int(rng.integers(len(SETTINGS)))
    if rng.random() < 0.1:
        c["peer"] = int(rng.integers(len(PEERS)))
    flip = rng.random(len(METRICS)) < 0.1
    c["inc"] = np.where(flip, 1 - c["inc"], c["inc"])
    mut = rng.random(len(METRICS)) < 0.2
    c["w"] = np.clip(c["w"] + mut * rng.normal(0, 0.2, len(METRICS)), -1, 1)
    return repair(c, rng)


def nsga2(S, rng, yts=None):
    pop = [repair(random_genome(rng), rng) for _ in range(POP)]
    res = [objectives(g, S, yts) for g in pop]
    for _ in range(GENS):
        objs = [(r[0], r[1]) for r in res]
        rank = np.empty(len(pop), int)
        crowd = np.empty(len(pop))
        for k, fr in enumerate(fronts(objs)):
            rank[fr] = k
            crowd[fr] = crowding(objs, fr)

        def pick():
            i, j = rng.integers(len(pop), size=2)
            return i if (rank[i], -crowd[i]) < (rank[j], -crowd[j]) else j
        kids = [vary(pop[pick()], pop[pick()], rng) for _ in range(POP)]
        allp, allr = pop + kids, res + [objectives(g, S, yts) for g in kids]
        objs = [(r[0], r[1]) for r in allr]
        keep = []
        for fr in fronts(objs):
            if len(keep) + len(fr) <= POP:
                keep += fr
            else:
                cd = crowding(objs, fr)
                keep += [fr[i] for i in np.argsort(-cd)[:POP - len(keep)]]
                break
        pop, res = [allp[i] for i in keep], [allr[i] for i in keep]
    return pop, res


def select(pop, res):
    """Pre-registered rule: feasible; within 90% of the best validity; fewest metrics; highest validity."""
    feas = [i for i, r in enumerate(res) if r[1] == 0]
    if not feas:
        return None
    best = max(res[i][2] for i in feas)
    near = [i for i in feas if res[i][2] >= (0.9 * best if best > 0 else best)]
    k = min(int(pop[i]["inc"].sum()) for i in near)
    cand = [i for i in near if pop[i]["inc"].sum() == k]
    return max(cand, key=lambda i: res[i][2])


def describe(gn):
    t, h = SETTINGS[gn["setting"]]
    parts = [f"{weights(gn)[j]:+.2f} x {LABEL[m]}" for j, m in enumerate(METRICS) if gn["inc"][j]]
    return f"turn >= {t} deg, window +-{h / 10:.1f} s, peers by {PEERS[gn['peer']]}: " + " ".join(parts)


# ---------------------------------------------------------------- holdout and rankings
def holdout(S, gn):
    s = S[gn["setting"]]
    te = s.test.copy()
    tr_x = s.composite(gn["peer"], weights(gn))
    x = s.composite(gn["peer"], weights(gn), s.ztest)
    te["comp"] = (x - tr_x.mean()) / tr_x.std()
    yv, X = patsy.dmatrices(f"separation ~ comp + {P.COVARS} + {P.FE}", te, return_type="dataframe")
    sub = te.loc[X.index]
    pt = DC.Partial(X.drop(columns=["comp"]).to_numpy(float), sub.nfl_id.to_numpy())
    r = DC.wild_cluster(pt, pt.M(yv.to_numpy(float).ravel()), pt.M(X["comp"].to_numpy(float)))
    r.update(n=len(sub), receivers=sub.nfl_id.nunique())
    return r


def player_scores(s, gn):
    d = s.all.copy()
    tr_x = s.composite(gn["peer"], weights(gn))
    d["comp"] = (s.composite(gn["peer"], weights(gn), s.zall) - tr_x.mean()) / tr_x.std()
    return d


def rank_intervals(d, col, rng, mass=None, resid_fit=None):
    ids = np.sort(d.nfl_id.unique())
    groups = [d.loc[d.nfl_id == i, col].to_numpy() for i in ids]
    point = np.array([g.mean() for g in groups])

    def transform(v):
        if mass is not None and resid_fit is None:
            return v * mass
        if resid_fit is not None:
            return v - (resid_fit[0] + resid_fit[1] * mass)
        return v
    score = transform(point)
    ranks = np.empty((B_RANK, len(ids)), int)
    for b in range(B_RANK):
        bs = transform(np.array([g[rng.integers(0, len(g), len(g))].mean() for g in groups]))
        ranks[b] = (-bs).argsort().argsort() + 1
    out = pd.DataFrame(dict(nfl_id=ids, score=score, n_plays=[len(g) for g in groups],
                            rank=(-score).argsort().argsort() + 1,
                            rank_lo=np.percentile(ranks, 2.5, 0), rank_hi=np.percentile(ranks, 97.5, 0)))
    return out.sort_values("rank")


# ---------------------------------------------------------------- figures
def fig_frontier(fr, sel_desc, bench_v, bench_r):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    v, rel, k = fr.validity.to_numpy(), fr.reliability.to_numpy(), fr.metrics.to_numpy()
    sel_i = int(np.flatnonzero(fr.description.to_numpy() == sel_desc)[0])
    for kk, color in zip([1, 2, 3, 4], [BLUE, AQUA, ORANGE, INK2]):
        m = k == kk
        if m.any():
            ax.scatter(rel[m], v[m], s=42, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                       label=f"{kk} metric{'s' if kk > 1 else ''}")
    ax.axvline(REL_MIN, color=INK2, lw=1, ls=":")
    ax.scatter([rel[sel_i]], [v[sel_i]], s=170, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=4,
               label="Selected (pre-registered rule)")
    ax.scatter([bench_r], [bench_v], s=90, marker="D", color=SURFACE, edgecolor=INK, linewidth=1.4, zorder=4,
               label="Stage 3 cut speed (benchmark)")
    ax.set_xlabel("Split-half reliability of the player score (2023–24)")
    ax.set_ylabel("Within-receiver effect on separation (yd per SD, 2023–24)")
    ax.set_title("Final NSGA-II population: validity against reliability, by number of metrics", loc="left",
                 fontsize=11)
    lo = min(v.min(), bench_v) - 0.02
    ax.set_ylim(lo, v.max() + 0.03)
    ax.text(REL_MIN, lo + 0.004, " reliability floor 0.60", color=INK2, fontsize=8.5, va="bottom")
    ax.legend(fontsize=8.5, loc="center right")
    mfp.save(fig, "ga_frontier.png")


def fig_rankings(comp, mom, names):
    fig, axes = plt.subplots(1, 2, figsize=(13, max(6, 0.26 * len(comp))), gridspec_kw={"wspace": 0.55})
    for ax, r, color, title in [(axes[0], comp, BLUE, "Cut composite (selected by the search)"),
                                (axes[1], mom, ORANGE, "Momentum through the cut (mass x cut speed)")]:
        r = r.sort_values("rank")
        y = np.arange(len(r))[::-1]
        ax.hlines(y, r.rank_lo, r.rank_hi, color=color, lw=2.2, alpha=0.55)
        ax.scatter(r["rank"], y, s=30, color=color, edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([names.get(i, str(i)) for i in r.nfl_id], fontsize=7.6)
        ax.set_xlabel("Rank (1 = best); line = 95% bootstrap interval")
        ax.set_xlim(0, len(r) + 1)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", fontsize=10.5)
    fig.subplots_adjust(top=0.95)
    fig.suptitle(f"{len(comp)} receivers with 30+ routes, 2023–25: ranks are wide; read the intervals, not the order",
                 x=0.07, ha="left", y=0.985, fontsize=11.5, color=INK)
    mfp.save(fig, "athlete_rankings.png")


def figures():
    """Redraw both figures from the saved results (no search needed)."""
    with open(os.path.join(cm.RESULTS_DIR, "ga_selection.json"), encoding="utf-8") as fh:
        s = json.load(fh)
    fr = pd.read_csv(os.path.join(cm.RESULTS_DIR, "ga_frontier.csv"))
    fig_frontier(fr, s["selected"], s["bench_validity"], s["bench_reliability"])
    tab = pd.read_csv(os.path.join(cm.RESULTS_DIR, "athlete_rankings.csv"))
    pick = lambda pre: tab[["nfl_id"] + [c for c in tab.columns if c.startswith(pre)]].rename(
        columns=lambda c: c.replace(pre, ""))
    fig_rankings(pick("cut_"), pick("momentum_"), dict(zip(tab.nfl_id, tab.player)))


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    data, base = build()
    S = {si: Setting(d, rng) for si, d in data.items()}
    print(f"prepared in {time.time() - t0:.0f} s", flush=True)

    pop, res = nsga2(S, rng)
    si = select(pop, res)
    sel = pop[si]
    sv, sr = res[si][2], res[si][3]
    bo = objectives({**BENCH, "inc": np.array(BENCH["inc"]), "w": np.array(BENCH["w"], float)}, S)
    print(f"search done in {time.time() - t0:.0f} s; selected: {describe(sel)}; validity {sv:+.3f}, rel {sr:.2f}",
          flush=True)

    # null: whole search on residuals permuted within receiver
    null = np.empty(N_NULL)
    nrng = np.random.default_rng(SEED + 1)
    for b in range(N_NULL):
        yts = {}
        for k, s in S.items():
            perm = np.arange(len(s.yt))
            for ix in s.members:
                perm[ix] = nrng.permutation(ix)
            yts[k] = s.pt.M(s.yt[perm])
        p_, r_ = nsga2(S, nrng, yts)
        j = select(p_, r_)
        null[b] = r_[j][2] if j is not None else 0.0
        print(f"  null search {b + 1}/{N_NULL}: selected validity {null[b]:+.3f}", flush=True)
    p_null = (1 + np.sum(null >= sv)) / (N_NULL + 1)

    ho = holdout(S, sel)
    bench_g = {**BENCH, "inc": np.array(BENCH["inc"]), "w": np.array(BENCH["w"], float)}
    hb = holdout(S, bench_g)

    # rankings (all seasons) with the selected definition; momentum and speed-for-size with its cut speed
    names = dict(zip(base.nfl_id, base.display_name))
    mass = dict(zip(base.nfl_id, base.combine_weight * cm.LB_TO_KG))
    s = S[sel["setting"]]
    d = player_scores(s, sel)
    comp = rank_intervals(d, "comp", np.random.default_rng(SEED + 2))
    dm = d[d.nfl_id.map(mass).notna()]
    ids = np.sort(dm.nfl_id.unique())
    m_arr = np.array([mass[i] for i in ids])
    mom = rank_intervals(dm, "apex", np.random.default_rng(SEED + 3), mass=m_arr)
    pm = dm.groupby("nfl_id").apex.mean().reindex(ids).to_numpy()
    fit = np.polyfit(m_arr, pm, 1)[::-1]
    sfs = rank_intervals(dm, "apex", np.random.default_rng(SEED + 4), mass=m_arr, resid_fit=fit)
    feats = pd.read_parquet(os.path.join(cm.OUT_DIR, "features_wr.parquet"))
    sep_mean = d.groupby("nfl_id").separation.mean()
    yprr = feats.set_index("nfl_id").loc[lambda f: f.routes >= 100, "yprr"]
    rho_sep = stats.spearmanr(comp.set_index("nfl_id").score, sep_mean.reindex(comp.nfl_id))
    yy = comp.set_index("nfl_id").score.to_frame().join(yprr, how="inner")
    rho_yprr = stats.spearmanr(yy.score, yy.yprr)

    tab = comp.rename(columns=lambda c: f"cut_{c}" if c != "nfl_id" else c)
    tab = tab.merge(mom.rename(columns=lambda c: f"momentum_{c}" if c != "nfl_id" else c), on="nfl_id", how="left")
    tab = tab.merge(sfs.rename(columns=lambda c: f"speed_for_size_{c}" if c != "nfl_id" else c), on="nfl_id",
                    how="left")
    tab.insert(1, "player", tab.nfl_id.map(names))
    tab["mass_kg"] = tab.nfl_id.map(mass)
    tab = tab.drop(columns=["momentum_n_plays", "speed_for_size_n_plays"])
    tab.to_csv(os.path.join(cm.RESULTS_DIR, "athlete_rankings.csv"), index=False)
    fr = pd.DataFrame([dict(description=describe(g), metrics=int(g["inc"].sum()), validity=r[2], reliability=r[3],
                            feasible=r[1] == 0) for g, r in zip(pop, res)]).sort_values("validity", ascending=False)
    fr.to_csv(os.path.join(cm.RESULTS_DIR, "ga_frontier.csv"), index=False)
    with open(os.path.join(cm.RESULTS_DIR, "ga_selection.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(selected=describe(sel), validity=sv, reliability=sr, bench_validity=bo[2],
                       bench_reliability=bo[3]), fh, indent=2)
    figures()

    L = ["Genetic-algorithm search for a receiver cut metric (results/ga_preregistration.md)", "",
         f"Training 2023-24, holdout 2025. NSGA-II population {POP}, {GENS} generations, seed {SEED}.",
         f"Selected: {describe(sel)}",
         f"  training validity {sv:+.3f} yd per SD, reliability {sr:.2f}",
         f"Benchmark (Stage 3 cut speed): training validity {bo[2]:+.3f}, reliability {bo[3]:.2f}",
         f"Null ({N_NULL} searches, residuals permuted within receiver): selected validity median "
         f"{np.median(null):+.3f}, 95th pct {np.percentile(null, 95):+.3f}; p = {p_null:.3f}",
         "",
         "HOLDOUT 2025 (primary), within receiver, wild cluster bootstrap:",
         f"  selected composite: {ho['coef']:+.3f} yd per SD [{ho['ci_wild'][0]:+.3f}, {ho['ci_wild'][1]:+.3f}], "
         f"p = {ho['p_wild']:.4f}; n = {ho['n']:,}, receivers = {ho['receivers']}",
         f"  benchmark cut speed: {hb['coef']:+.3f} [{hb['ci_wild'][0]:+.3f}, {hb['ci_wild'][1]:+.3f}], "
         f"p = {hb['p_wild']:.4f}; n = {hb['n']:,}",
         "",
         f"RANKINGS ({len(comp)} receivers, 30+ plays, 2023-25; 95% bootstrap rank intervals)",
         f"  median rank-interval width: cut composite {np.median(comp.rank_hi - comp.rank_lo):.0f} places, "
         f"momentum {np.median(mom.rank_hi - mom.rank_lo):.0f}, speed for size {np.median(sfs.rank_hi - sfs.rank_lo):.0f}",
         f"  between-player check: Spearman(score, mean separation) = {rho_sep.statistic:+.2f} (p = {rho_sep.pvalue:.3f}); "
         f"Spearman(score, YPRR, 100+ routes) = {rho_yprr.statistic:+.2f} (p = {rho_yprr.pvalue:.3f}, N = {len(yy)})",
         "  Top 10, cut composite: " + "; ".join(f"{names.get(r.nfl_id)} {int(r.rank)} [{r.rank_lo:.0f}-{r.rank_hi:.0f}]"
                                                for r in comp.head(10).itertuples()),
         "  Top 10, momentum: " + "; ".join(f"{names.get(r.nfl_id)} {r.score:.0f} kg m/s [{r.rank_lo:.0f}-{r.rank_hi:.0f}]"
                                           for r in mom.head(10).itertuples()),
         "  Top 10, speed for size: " + "; ".join(f"{names.get(r.nfl_id)} {r.score:+.2f} m/s [{r.rank_lo:.0f}-{r.rank_hi:.0f}]"
                                                 for r in sfs.head(10).itertuples()),
         f"Run time {time.time() - t0:.0f} s"]
    txt = "\n".join(L)
    with open(os.path.join(cm.RESULTS_DIR, "ga_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    figures() if "--figures" in sys.argv else main()
