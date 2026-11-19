"""
game_cod_features.py
In-game change-of-direction cuts from regular-season game tracking, detected and
measured exactly as at the Combine (cod_features.py): Savitzky-Golay smoothing,
cut = slowest frame of a run turning >= 45 degrees, apex/entry/exit speed and
peak, braking and turning force (F = m a with the player's Combine weight).

Only the active phase counts: from the snap to the throw on pass plays (at most
4 s), or the first 3 s after the snap otherwise.

Streams the tracking in chunks (from DATA_DIR, or the zip named by NFL_BDB_ZIP),
so it never holds a season in memory. Writes data/game_cuts.parquet.
"""
import os

import numpy as np
import pandas as pd

import common as cm
import cod_features as cf

PASS_CAP, RUN_WINDOW = 40, 30     # frames after the snap


def play_cuts(x, y, ev, mass):
    """Cuts in one player-play. Returns a list of dicts (frame relative to snap)."""
    snaps = np.flatnonzero(ev == "ball_snap")
    if len(snaps) == 0 or len(x) < cf.WINDOW + 2:
        return []
    s0 = snaps[0]
    passes = np.flatnonzero(ev[s0:] == "pass_forward")
    is_pass = len(passes) > 0
    end = s0 + (min(passes[0], PASS_CAP) if is_pass else RUN_WINDOW)
    g = {"x": x, "y": y}
    vx, vy, v, a, a_t, a_n = cf.kinematics(pd.DataFrame(g))
    out, n = [], len(v)
    for k, angle in cf.find_cuts(v, cf.turn_angles(vx, vy)):
        if not (s0 <= k <= end):
            continue
        lo, hi = max(0, k - cf.HALF), min(n, k + cf.HALF + 1)
        pre = v[max(0, k - 2 * cf.HALF):k + 1]
        out.append(dict(
            t_after_snap=(k - s0) / 10, turn_deg=angle, entry=pre.max(), apex=v[k],
            exit05=v[min(n - 1, k + cf.HALF)],
            retention=v[k] / pre.max() if pre.max() > 0 else np.nan,
            f_peak=mass * a[lo:hi].max(), f_brake=mass * max(0.0, -a_t[lo:hi].min()),
            f_turn=mass * a_n[lo:hi].max(), pass_play=is_pass))
    return out


def process_frame(df, mass, rows):
    key = df.game_id.to_numpy() * 100_000_000 + df.play_id.to_numpy() * 100_000 + df.nfl_id.to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    ends = np.r_[starts[1:], len(df)]
    x, y = df.x.to_numpy(np.float64), df.y.to_numpy(np.float64)
    ev = df.event.to_numpy(object)
    gid, pid, nid = df.game_id.to_numpy(), df.play_id.to_numpy(), df.nfl_id.to_numpy()
    for s, e in zip(starts, ends):
        m = mass.get(int(nid[s]))
        if m is None:
            continue
        for r in play_cuts(x[s:e], y[s:e], ev[s:e], m):
            r.update(game_id=int(gid[s]), play_id=int(pid[s]), nfl_id=int(nid[s]))
            rows.append(r)


def main():
    base = cm.player_base()
    base = base[base.combine_weight.notna()]
    mass = dict(zip(base.nfl_id.astype(int), base.combine_weight * cm.LB_TO_KG))
    games = pd.read_csv(cm.path("games.csv"))
    reg = set(games.loc[games.season_type == "REG", "game_id"])
    season = dict(zip(games.game_id, games.season))
    rows = []
    cols = ["game_id", "play_id", "nfl_id", "x", "y", "event"]
    dtype = {"game_id": "int64", "play_id": "int64", "nfl_id": "int64", "x": "float32", "y": "float32", "event": "object"}
    for year in (2023, 2024, 2025):
        carry = None
        with cm.open_game_tracking(year) as f:
            for ch in pd.read_csv(f, usecols=cols, dtype=dtype, chunksize=1_000_000):
                ch = ch[ch.game_id.isin(reg)]
                if carry is not None:
                    ch = pd.concat([carry, ch], ignore_index=True)
                if ch.empty:
                    continue
                last = (ch.game_id == ch.game_id.iloc[-1]) & (ch.play_id == ch.play_id.iloc[-1]) & (ch.nfl_id == ch.nfl_id.iloc[-1])
                carry = ch[last]
                process_frame(ch[~last].reset_index(drop=True), mass, rows)
        if carry is not None and len(carry):
            process_frame(carry.reset_index(drop=True), mass, rows)
        print(f"season {year}: {len(rows):,} cuts so far", flush=True)
    out = pd.DataFrame(rows)
    out["season"] = out.game_id.map(season).astype("int16")
    for c in ["t_after_snap", "turn_deg", "entry", "apex", "exit05", "retention", "f_peak", "f_brake", "f_turn"]:
        out[c] = out[c].astype("float32")
    os.makedirs(cm.OUT_DIR, exist_ok=True)
    out.to_parquet(os.path.join(cm.OUT_DIR, "game_cuts.parquet"), index=False)
    print(f"{len(out):,} cuts, {out.nfl_id.nunique()} players, {out.groupby(['game_id', 'play_id', 'nfl_id']).ngroups:,} player-plays")


if __name__ == "__main__":
    main()
