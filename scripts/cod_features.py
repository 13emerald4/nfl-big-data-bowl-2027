"""
cod_features.py
Change-of-direction (COD) speed and force metrics from Combine tracking.
Uses drill data and body mass only; no career or game outcome is read here,
so every definition is fixed before any outcome is looked at.

Idea 1  Speed held through the cut, relative to peers (receiver route drills).
Idea 2  Horizontal force produced at each instantaneous speed through the cut,
        relative to body size (short shuttle and three-cone, every position).

Kinematics. x, y (yards) are smoothed with a Savitzky-Golay filter (7 frames,
quadratic), whose derivatives give the velocity and acceleration vectors. The
7-frame window matches the provider's own s and a columns best (median per-rep
correlation 1.00 for speed, 0.97 for |a|). Acceleration is split into
  tangential a_t = (v . a)/|v|   (+ speeding up, - braking)
  normal     a_n = |v x a|/|v|   (turning; equals v^2 / radius)
and horizontal force is F = m |a| with m the Combine weight in kg. Vertical
force (gravity, jumping) is not observable from x, y.

COD event. Turn angle at frame i is the angle between the velocity 0.5 s
before and 0.5 s after. Frames with a turn angle of at least 45 degrees form
runs; each run is one cut, located at its slowest frame (the plant).

Writes data/cod_events.parquet, data/features_cod.parquet and
results/cod_reliability.txt.
"""
import os

import numpy as np
import pandas as pd
import polars as pl
from scipy.signal import savgol_filter

import common as cm

YD_TO_M = 0.9144
WINDOW = 7          # Savitzky-Golay frames (0.7 s)
HALF = 5            # frames either side for the turn angle and force window (0.5 s)
MIN_TURN = 45.0     # degrees
ANGLE_BINS = [45, 90, 135, 180.01]
SPEED_BINS = [0, 1, 2, 3, 4, np.inf]   # m/s, for the force-velocity profile

WR_ROUTE_DRILLS = [
    "DAGGER_ROUTE_LEFT", "SLANT_ROUTE_LEFT", "CURL_ROUTE_RIGHT", "SLOT_STRIKE_ROUTE_LEFT",
    "SLOT_SAIL_ROUTE_RIGHT", "COMEBACK_ROUTE_RIGHT", "POST_CORNER_ROUTE_RIGHT",
    "POST_CORNER_ROUTE_LEFT", "SPEED_OUT_ROUTE_LEFT", "SPEED_OUT_LEFT",
]
DRILL_ALIAS = {"SPEED_OUT_LEFT": "SPEED_OUT_ROUTE_LEFT"}
AGILITY_DRILLS = {"SHORT_SHUTTLE": 2, "THREE_CONE_DRILL": 3}   # minimum cuts for a complete rep

POS_GROUP = {
    "WR": "WR", "TE": "TE", "RB": "RB", "FB": "RB", "QB": "QB",
    "T": "OL", "G": "OL", "C": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL",
    "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB": "LB",
    "CB": "DB", "SS": "DB", "FS": "DB", "DB": "DB", "S": "DB",
}


def kinematics(g):
    """Smoothed velocity/acceleration (m/s, m/s^2) for one rep, ordered by time."""
    x, y = g["x"].to_numpy() * YD_TO_M, g["y"].to_numpy() * YD_TO_M
    vx = savgol_filter(x, WINDOW, 2, deriv=1, delta=0.1)
    vy = savgol_filter(y, WINDOW, 2, deriv=1, delta=0.1)
    ax = savgol_filter(x, WINDOW, 2, deriv=2, delta=0.1)
    ay = savgol_filter(y, WINDOW, 2, deriv=2, delta=0.1)
    v = np.hypot(vx, vy)
    safe = np.maximum(v, 1e-6)
    a_t = (vx * ax + vy * ay) / safe
    a_n = np.abs(vx * ay - vy * ax) / safe
    return vx, vy, v, np.hypot(ax, ay), a_t, a_n


def turn_angles(vx, vy):
    """Angle (deg) between the velocity HALF frames before and after each frame."""
    n = len(vx)
    ang = np.full(n, np.nan)
    i = np.arange(HALF, n - HALF)
    b = np.stack([vx[i - HALF], vy[i - HALF]], 1)
    f = np.stack([vx[i + HALF], vy[i + HALF]], 1)
    nb, nf = np.linalg.norm(b, axis=1), np.linalg.norm(f, axis=1)
    ok = (nb > 0.3) & (nf > 0.3)
    cos = np.clip((b * f).sum(1) / np.maximum(nb * nf, 1e-9), -1, 1)
    ang[i[ok]] = np.degrees(np.arccos(cos[ok]))
    return ang


def find_cuts(v, ang):
    """Index of the slowest frame in each run of frames turning >= MIN_TURN."""
    hot = np.nan_to_num(ang) >= MIN_TURN
    cuts, i, n = [], 0, len(v)
    while i < n:
        if hot[i]:
            j = i
            while j + 1 < n and hot[j + 1]:
                j += 1
            k = i + int(np.argmin(v[i:j + 1]))
            if cuts and k - cuts[-1][0] < HALF:          # merge runs closer than 0.5 s
                if v[k] < v[cuts[-1][0]]:
                    cuts[-1] = (k, max(cuts[-1][1], np.nanmax(ang[i:j + 1])))
            else:
                cuts.append((k, float(np.nanmax(ang[i:j + 1]))))
            i = j + 1
        else:
            i += 1
    return cuts


def rep_events(g, mass):
    """One row per cut in a rep, plus that rep's force frames around each cut."""
    vx, vy, v, a, a_t, a_n = kinematics(g)
    cuts = find_cuts(v, turn_angles(vx, vy))
    rows, frames = [], []
    n = len(v)
    for k, angle in cuts:
        lo, hi = max(0, k - HALF), min(n, k + HALF + 1)
        pre = v[max(0, k - 2 * HALF):k + 1]
        post = v[k:min(n, k + 2 * HALF + 1)]
        rows.append(dict(
            frame=k, turn_deg=angle,
            entry=pre.max(), apex=v[k],
            exit05=v[min(n - 1, k + HALF)], exit_max=post.max(),
            retention=v[k] / pre.max() if pre.max() > 0 else np.nan,
            f_peak=mass * a[lo:hi].max(),
            f_brake=mass * max(0.0, -a_t[lo:hi].min()),
            f_turn=mass * a_n[lo:hi].max(),
            complete=(k - 2 * HALF >= 0) and (k + 2 * HALF < n),
        ))
        frames.append(pd.DataFrame({"v": v[lo:hi], "force": mass * a[lo:hi]}))
    return rows, frames


def build_events(ct, base):
    mass = dict(zip(base.nfl_id, base.combine_weight * cm.LB_TO_KG))
    keep = WR_ROUTE_DRILLS + list(AGILITY_DRILLS)
    sub = ct.filter(pl.col("drill_name").is_in(keep) & pl.col("nfl_id").is_in(list(mass)))
    events, fv = [], []
    for (eid,), g in sub.group_by(["event_id"], maintain_order=True):
        if g.height < WINDOW + 2:
            continue
        pid, drill = g["nfl_id"][0], g["drill_name"][0]
        rows, frames = rep_events(g, mass[pid])
        if drill in AGILITY_DRILLS and len(rows) < AGILITY_DRILLS[drill]:
            continue                                    # truncated or aborted rep
        for r in rows:
            r.update(nfl_id=pid, event_id=eid, drill=DRILL_ALIAS.get(drill, drill),
                     attempt=g["attempt"][0], draft_year=g["draft_year"][0])
        events += rows
        for f in frames:
            f["nfl_id"], f["drill"], f["event_id"] = pid, drill, eid
        fv += frames
    return pd.DataFrame(events), pd.concat(fv, ignore_index=True)


def zscore(s):
    return (s - s.mean()) / s.std(ddof=1)


def idea1_scores(ev):
    """Peer-relative cut speed for receivers: z within drill x turn-angle bin,
    averaged within each drill, then across drills (>= 3 drills)."""
    wr = ev[ev.drill.isin(WR_ROUTE_DRILLS) & ev.complete].copy()
    wr["angle_bin"] = pd.cut(wr.turn_deg, ANGLE_BINS, right=False)
    grp = wr.groupby(["drill", "angle_bin"], observed=True)
    wr = wr[grp["apex"].transform("size") >= 20].copy()
    grp = wr.groupby(["drill", "angle_bin"], observed=True)
    for col in ["apex", "retention", "exit05"]:
        wr[f"{col}_z"] = grp[col].transform(zscore)
    per_drill = wr.groupby(["nfl_id", "drill"])[["apex_z", "retention_z", "exit05_z", "apex", "entry"]].mean().reset_index()
    out = per_drill.groupby("nfl_id").agg(
        cod_speed_z=("apex_z", "mean"), cod_retention_z=("retention_z", "mean"),
        cod_exit_z=("exit05_z", "mean"), cod_apex_ms=("apex", "mean"),
        cod_entry_ms=("entry", "mean"), cod_drills=("drill", "size"))
    out["cod_cuts"] = wr.groupby("nfl_id").size()
    return out[out.cod_drills >= 3].reset_index(), wr, per_drill


def idea1_three_cone(ev, base):
    """Speed held through the three-cone's rounded cuts (45-90 degrees; the
    others are ~180-degree plants), every position: mean apex speed per player,
    z-scored within position group (peers)."""
    tc = ev[(ev.drill == "THREE_CONE_DRILL") & ev.turn_deg.between(45, 90)]
    out = tc.groupby("nfl_id").agg(tc_cut_speed_ms=("apex", "mean"), tc_cuts=("apex", "size")).reset_index()
    out = out.merge(base[["nfl_id", "nfl_position"]], on="nfl_id")
    out["grp"] = out.nfl_position.map(POS_GROUP).fillna("OTHER")
    out = out[out.groupby("grp").nfl_id.transform("size") >= 8].copy()
    out["tc_cut_speed_z"] = out.groupby("grp").tc_cut_speed_ms.transform(zscore)
    return out.drop(columns=["nfl_position", "grp"])


def residual_on_mass(df, col, mass):
    X = np.column_stack([np.ones(len(df)), df[mass]])
    beta, *_ = np.linalg.lstsq(X, df[col], rcond=None)
    return df[col] - X @ beta, beta


def idea2_scores(ev, fv, base):
    """Force at the cut (shuttle + three-cone): absolute N, N/kg and force over
    expected for body mass, standardised within drill, averaged across drills."""
    ag = ev[ev.drill.isin(AGILITY_DRILLS)].copy()
    per = ag.groupby(["nfl_id", "drill"]).agg(f_peak=("f_peak", "median"), f_brake=("f_brake", "median"),
                                             f_turn=("f_turn", "median"), apex=("apex", "median")).reset_index()
    per = per.merge(base[["nfl_id", "combine_weight"]], on="nfl_id")
    per["mass_kg"] = per.combine_weight * cm.LB_TO_KG
    slopes = {}
    for d, g in per.groupby("drill"):
        for col in ["f_peak", "f_brake", "f_turn"]:
            r, beta = residual_on_mass(g, col, "mass_kg")
            per.loc[g.index, f"{col}_oe_z"] = zscore(r)
            slopes[(d, col)] = beta
        per.loc[g.index, "f_peak_z"] = zscore(g.f_peak)
        per.loc[g.index, "f_per_kg_z"] = zscore(g.f_peak / g.mass_kg)
    out = per.groupby("nfl_id").agg(
        force_oe_z=("f_peak_oe_z", "mean"), brake_oe_z=("f_brake_oe_z", "mean"),
        turn_oe_z=("f_turn_oe_z", "mean"), force_abs_z=("f_peak_z", "mean"),
        force_per_kg_z=("f_per_kg_z", "mean"), force_n=("f_peak", "mean"),
        agility_drills=("drill", "size")).reset_index()

    # Force-velocity profile: 90th-percentile force in each speed band, pooled
    # over every frame within 0.5 s of a cut in either agility drill.
    f = fv[fv.drill.isin(AGILITY_DRILLS)].copy()
    f["band"] = pd.cut(f.v, SPEED_BINS, right=False, labels=["0-1", "1-2", "2-3", "3-4", "4+"])
    prof = (f.groupby(["nfl_id", "band"], observed=True)
            .agg(p90=("force", lambda s: s.quantile(0.9)), n=("force", "size"), v=("v", "mean")).reset_index())
    prof = prof[prof.n >= 3]
    fits = []
    for pid, g in prof.groupby("nfl_id"):
        if len(g) >= 3:
            slope, f0 = np.polyfit(g.v, g.p90, 1)
            fits.append(dict(nfl_id=pid, fv_f0=f0, fv_slope=slope))
    wide = prof.pivot(index="nfl_id", columns="band", values="p90")
    wide.columns = [f"force_{c.replace('-', '_').replace('+', 'plus')}_ms" for c in wide.columns]
    out = out.merge(pd.DataFrame(fits), on="nfl_id", how="left").merge(wide.reset_index(), on="nfl_id", how="left")
    return out, per, prof, slopes


def reliability(wr, per_drill, per, ev):
    lines = []
    # Idea 1: odd vs even drills (alphabetical within player), Spearman-Brown corrected.
    pdd = per_drill.sort_values(["nfl_id", "drill"]).copy()
    pdd["half"] = pdd.groupby("nfl_id").cumcount() % 2
    h = pdd.groupby(["nfl_id", "half"])["apex_z"].mean().unstack().dropna()
    r = h[0].corr(h[1])
    lines.append(f"Idea 1 cut-speed z, odd vs even drills: r = {r:.2f} (N = {len(h)}), "
                 f"Spearman-Brown full-length reliability = {2 * r / (1 + r):.2f}")
    for col in ["retention_z", "exit05_z"]:
        h = pdd.assign(**{col: per_drill.set_index(["nfl_id", "drill"]).loc[list(zip(pdd.nfl_id, pdd.drill)), col].to_numpy()})
        h = h.groupby(["nfl_id", "half"])[col].mean().unstack().dropna()
        r = h[0].corr(h[1])
        lines.append(f"Idea 1 {col}, odd vs even drills: r = {r:.2f}, Spearman-Brown = {2 * r / (1 + r):.2f}")
    rep = wr.groupby(["nfl_id", "drill", "event_id"])["apex_z"].mean().reset_index()
    rep["k"] = rep.groupby(["nfl_id", "drill"]).cumcount()
    w = rep[rep.k < 2].pivot_table(index=["nfl_id", "drill"], columns="k", values="apex_z").dropna()
    lines.append(f"Idea 1 cut-speed z, rep 1 vs rep 2 of the same drill: r = {w[0].corr(w[1]):.2f} (N = {len(w)} pairs)")
    c = per_drill.pivot(index="nfl_id", columns="drill", values="apex_z").corr(min_periods=30).to_numpy()
    lines.append(f"Idea 1 cut-speed z, mean correlation between two different drills: r = "
                 f"{np.nanmean(c[~np.eye(len(c), dtype=bool)]):.2f}")
    # Idea 2: shuttle vs three-cone, and attempt 1 vs 2 within a drill.
    w = per.pivot(index="nfl_id", columns="drill", values="f_peak_oe_z").dropna()
    lines.append(f"Idea 2 force over expected, shuttle vs three-cone: r = {w.iloc[:, 0].corr(w.iloc[:, 1]):.2f} (N = {len(w)})")
    w = per.pivot(index="nfl_id", columns="drill", values="f_peak").dropna()
    lines.append(f"Idea 2 absolute force, shuttle vs three-cone: r = {w.iloc[:, 0].corr(w.iloc[:, 1]):.2f}")
    ag = ev[ev.drill.isin(AGILITY_DRILLS)]
    rep = ag.groupby(["nfl_id", "drill", "event_id"]).agg(f=("f_peak", "median"), att=("attempt", "first")).reset_index()
    rep["k"] = rep.groupby(["nfl_id", "drill"]).cumcount()
    w = rep[rep.k < 2].pivot_table(index=["nfl_id", "drill"], columns="k", values="f").dropna()
    lines.append(f"Idea 2 absolute force, rep 1 vs rep 2 of the same drill: r = {w[0].corr(w[1]):.2f} (N = {len(w)} pairs)")
    rep = rep.merge(per[["nfl_id", "mass_kg"]].drop_duplicates(), on="nfl_id")
    for d, g in rep.groupby("drill"):
        rep.loc[g.index, "foe"], _ = residual_on_mass(g, "f", "mass_kg")
    w = rep[rep.k < 2].pivot_table(index=["nfl_id", "drill"], columns="k", values="foe").dropna()
    lines.append(f"Idea 2 force over expected, rep 1 vs rep 2 of the same drill: r = {w[0].corr(w[1]):.2f}")
    lines.append(f"Idea 2 body mass explains {np.corrcoef(per.f_peak, per.mass_kg)[0, 1] ** 2:.0%} of peak-force variance")
    return lines


def main():
    ct = cm.combine_tracking()
    base = cm.player_base()
    base = base[base.combine_weight.notna()]
    ev, fv = build_events(ct, base)
    s1, wr, per_drill = idea1_scores(ev)
    s2, per, prof, slopes = idea2_scores(ev, fv, base)

    feats = base[["nfl_id", "display_name", "nfl_position", "draft_year", "combine_weight"]].copy()
    feats["pos_group"] = feats.nfl_position.map(POS_GROUP).fillna("OTHER")
    s3 = idea1_three_cone(ev, base)
    feats = (feats.merge(s1, on="nfl_id", how="left").merge(s2, on="nfl_id", how="left")
             .merge(s3, on="nfl_id", how="left"))
    feats = feats[feats.cod_speed_z.notna() | feats.force_oe_z.notna()]

    os.makedirs(cm.OUT_DIR, exist_ok=True)
    ev.to_parquet(os.path.join(cm.OUT_DIR, "cod_events.parquet"), index=False)
    feats.to_parquet(os.path.join(cm.OUT_DIR, "features_cod.parquet"), index=False)

    lines = ["Change-of-direction metrics (drill data only; no outcomes read)", ""]
    lines.append(f"Cuts detected: {len(ev):,} in {ev.event_id.nunique():,} reps")
    lines.append(ev.groupby("drill").agg(reps=("event_id", "nunique"), cuts=("frame", "size"),
                                         cuts_per_rep=("frame", lambda s: len(s)),
                                         median_turn=("turn_deg", "median"), median_apex_ms=("apex", "median"))
                 .assign(cuts_per_rep=lambda d: (d.cuts / d.reps).round(2)).round(2).to_string())
    lines.append("")
    lines.append(f"Idea 1 receivers scored: {s1.shape[0]} (>= 3 route drills); median {s1.cod_drills.median():.0f} drills, "
                 f"{s1.cod_cuts.median():.0f} cuts each")
    lines.append(f"Idea 1 three-cone rounded-cut speed scored: {s3.shape[0]} players (all positions)")
    lines.append(f"Idea 2 players scored: {s2.shape[0]} (shuttle and/or three-cone); "
                 f"{(s2.agility_drills == 2).sum()} ran both")
    for (d, col), (a0, b) in sorted(slopes.items()):
        if col == "f_peak":
            lines.append(f"  {d}: peak force = {a0:.0f} N + {b:.2f} N per kg of body mass")
    lines.append("")
    lines.append("Force-velocity profile, median across players (N, 90th percentile within 0.5 s of a cut):")
    lines.append(prof.groupby("band", observed=True).agg(players=("nfl_id", "nunique"), median_force_n=("p90", "median"))
                 .round(0).to_string())
    lines.append("")
    lines += reliability(wr, per_drill, per, ev)
    txt = "\n".join(lines)
    os.makedirs(cm.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(cm.RESULTS_DIR, "cod_reliability.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
