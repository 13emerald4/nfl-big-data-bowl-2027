"""
ga_cut_features.py
Cut metrics for results/ga_preregistration.md. One streaming pass over game
tracking; for every cohort WR/TE route on a regular-season pass play it finds
the last cut before the throw under nine detection settings (minimum turn
{30, 45, 60} deg x window {3, 5, 7} frames) and measures:

  apex       speed at the slowest frame of the cut (m/s)
  entry      peak speed in the 1.0 s before the cut
  retention  apex / entry
  reaccel    mean tangential acceleration over the 0.5 s after the cut (m/s^2)
  recover    time until speed is back to 90% of entry (s, capped at 1.5)
  brake      peak deceleration within +-window (m/s^2)
  turnacc    peak normal acceleration within +-window (m/s^2)

Smoothing is the validated 7-frame Savitzky-Golay filter (cod_features.kinematics).
Writes data/ga_cuts.parquet.
"""
import os

import numpy as np
import pandas as pd

import common as cm
import cod_features as cf

PASS_CAP = 40
TURNS, HALVES = (30.0, 45.0, 60.0), (3, 5, 7)
ENTRY_FRAMES, REACCEL_FRAMES, RECOVER_CAP = 10, 5, 15


def turn_angles(vx, vy, half):
    n = len(vx)
    ang = np.full(n, np.nan)
    if n <= 2 * half:
        return ang
    i = np.arange(half, n - half)
    b = np.stack([vx[i - half], vy[i - half]], 1)
    f = np.stack([vx[i + half], vy[i + half]], 1)
    nb, nf = np.linalg.norm(b, axis=1), np.linalg.norm(f, axis=1)
    ok = (nb > 0.3) & (nf > 0.3)
    cos = np.clip((b * f).sum(1) / np.maximum(nb * nf, 1e-9), -1, 1)
    ang[i[ok]] = np.degrees(np.arccos(cos[ok]))
    return ang


def find_cuts(v, ang, min_turn, half):
    """Same rule as cod_features.find_cuts, with the threshold and window as arguments."""
    hot = np.nan_to_num(ang) >= min_turn
    cuts, i, n = [], 0, len(v)
    while i < n:
        if hot[i]:
            j = i
            while j + 1 < n and hot[j + 1]:
                j += 1
            k = i + int(np.argmin(v[i:j + 1]))
            if cuts and k - cuts[-1][0] < half:
                if v[k] < v[cuts[-1][0]]:
                    cuts[-1] = (k, max(cuts[-1][1], np.nanmax(ang[i:j + 1])))
            else:
                cuts.append((k, float(np.nanmax(ang[i:j + 1]))))
            i = j + 1
        else:
            i += 1
    return cuts


def play_metrics(x, y, ev):
    snaps = np.flatnonzero(ev == "ball_snap")
    if len(snaps) == 0 or len(x) < cf.WINDOW + 2:
        return []
    s0 = snaps[0]
    passes = np.flatnonzero(ev[s0:] == "pass_forward")
    if len(passes) == 0:
        return []
    end = s0 + min(passes[0], PASS_CAP)
    vx, vy, v, _, a_t, a_n = cf.kinematics(pd.DataFrame({"x": x, "y": y}))
    n, out = len(v), []
    for half in HALVES:
        ang = turn_angles(vx, vy, half)
        for mt in TURNS:
            cuts = [(k, a) for k, a in find_cuts(v, ang, mt, half) if s0 <= k <= end]
            if not cuts:
                continue
            k, a = cuts[-1]
            entry = v[max(0, k - ENTRY_FRAMES):k + 1].max()
            after = v[k + 1:min(n, k + 1 + RECOVER_CAP)]
            back = np.flatnonzero(after >= 0.9 * entry)
            lo, hi = max(0, k - half), min(n, k + half + 1)
            ra = a_t[k + 1:min(n, k + 1 + REACCEL_FRAMES)]
            out.append(dict(min_turn=int(mt), half=half, t_after_snap=(k - s0) / 10, turn_deg=a, entry=entry,
                            apex=v[k], retention=v[k] / entry if entry > 0 else np.nan,
                            reaccel=ra.mean() if len(ra) else np.nan,
                            recover=(back[0] + 1) / 10 if len(back) else RECOVER_CAP / 10,
                            brake=max(0.0, -a_t[lo:hi].min()), turnacc=a_n[lo:hi].max()))
    return out


def process(df, keys, rows):
    key = df.game_id.to_numpy() * 100_000_000 + df.play_id.to_numpy() * 100_000 + df.nfl_id.to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    ends = np.r_[starts[1:], len(df)]
    x, y, ev = df.x.to_numpy(np.float64), df.y.to_numpy(np.float64), df.event.to_numpy(object)
    gid, pid, nid = df.game_id.to_numpy(), df.play_id.to_numpy(), df.nfl_id.to_numpy()
    for a, b in zip(starts, ends):
        k3 = (int(gid[a]), int(pid[a]), int(nid[a]))
        if k3 not in keys:
            continue
        for r in play_metrics(x[a:b], y[a:b], ev[a:b]):
            r.update(game_id=k3[0], play_id=k3[1], nfl_id=k3[2])
            rows.append(r)


def main():
    base = cm.player_base()
    grp = base.set_index("nfl_id").nfl_position.map(cf.POS_GROUP)
    pp = cm.player_play_regular_season(usecols=["game_id", "play_id", "nfl_id", "route_ran"])
    pp = pp[pp.route_ran.notna() & pp.nfl_id.map(grp).isin(["WR", "TE"])]
    keys = set(zip(pp.game_id.astype(int), pp.play_id.astype(int), pp.nfl_id.astype(int)))
    print(f"{len(keys):,} WR/TE route player-plays", flush=True)
    del pp
    games = pd.read_csv(cm.path("games.csv"))
    reg = set(games.loc[games.season_type == "REG", "game_id"])
    cols = ["game_id", "play_id", "nfl_id", "x", "y", "event"]
    dtype = {"game_id": "int64", "play_id": "int64", "nfl_id": "int64", "x": "float32", "y": "float32",
             "event": "object"}
    rows = []
    for year in (2023, 2024, 2025):
        carry = None
        with cm.open_game_tracking(year) as f:
            for ch in pd.read_csv(f, usecols=cols, dtype=dtype, chunksize=1_000_000):
                ch = ch[ch.game_id.isin(reg)]
                if carry is not None:
                    ch = pd.concat([carry, ch], ignore_index=True)
                if ch.empty:
                    continue
                last = (ch.game_id == ch.game_id.iloc[-1]) & (ch.play_id == ch.play_id.iloc[-1]) & \
                       (ch.nfl_id == ch.nfl_id.iloc[-1])
                carry = ch[last]
                process(ch[~last].reset_index(drop=True), keys, rows)
        if carry is not None and len(carry):
            process(carry.reset_index(drop=True), keys, rows)
        print(f"season {year}: {len(rows):,} rows", flush=True)
    out = pd.DataFrame(rows)
    for c in ["t_after_snap", "turn_deg", "entry", "apex", "retention", "reaccel", "recover", "brake", "turnacc"]:
        out[c] = out[c].astype("float32")
    out.to_parquet(os.path.join(cm.OUT_DIR, "ga_cuts.parquet"), index=False)
    print(out.groupby(["min_turn", "half"]).size().to_string())


if __name__ == "__main__":
    main()
