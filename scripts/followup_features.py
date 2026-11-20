"""
followup_features.py
One streaming pass over game tracking that builds the treatments for
results/followup_preregistration.md:

  data/fu_player_play.parquet  every regular-season player-play: peak speed,
                               speed at pass_outcome_caught / pass_arrived,
                               OL get-off (time from the snap to 1.0 m/s, cap 2 s)
                               and displacement 1.0 s after the snap (T2, T4, T7)
  data/fu_shoulders.parquet    Stage 3 receiver plays: early shoulder turn
                               before the last cut (T5)
  data/fu_paths.parquet        snap-to-throw paths of Stage 3 receivers and of
                               cohort MAN defenders on the same plays (T6)

Needs data/play_level.parquet (play_level_fe.py).
"""
import os

import numpy as np
import pandas as pd

import common as cm

PATH_FRAMES = 41            # snap .. snap + 4.0 s
GETOFF_SPEED, GETOFF_CAP = 1.0 / 0.9144, 20   # provider s is yd/s; threshold 1.0 m/s


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def circ_mean(deg):
    r = np.deg2rad(deg)
    return np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))


def first(ev, name, start=0):
    i = np.flatnonzero(ev[start:] == name)
    return start + i[0] if len(i) else -1


def shoulders(o, dr, k):
    """Signed angle between orientation 0.3 s before the cut and the stem direction, + = toward the break."""
    n = len(o)
    if k - 10 < 0 or k + 4 > n - 1:
        return None
    stem = circ_mean(dr[k - 10:k - 4])          # 1.0 .. 0.5 s before the cut
    post = circ_mean(dr[k + 3:min(n, k + 7)])   # 0.3 .. 0.6 s after
    turn = wrap(post - stem)
    if abs(turn) < 1e-6:
        return None
    lead = wrap(o[k - 3] - stem) * np.sign(turn)
    return dict(shoulder_lead=lead, o_stem_gap=wrap(o[k - 6] - stem) * np.sign(turn), turn_signed=turn)


def process(df, cut_t, path_keys, pp_rows, sh_rows, path_rows):
    key = df.game_id.to_numpy() * 100_000_000 + df.play_id.to_numpy() * 100_000 + df.nfl_id.to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    ends = np.r_[starts[1:], len(df)]
    X, Y, S = df.x.to_numpy(np.float64), df.y.to_numpy(np.float64), df.s.to_numpy(np.float64)
    O, D = df.o.to_numpy(np.float64), df.dir.to_numpy(np.float64)
    EV = df.event.to_numpy(object)
    gid, pid, nid = df.game_id.to_numpy(), df.play_id.to_numpy(), df.nfl_id.to_numpy()
    for a, b in zip(starts, ends):
        ev, s = EV[a:b], S[a:b]
        k3 = (int(gid[a]), int(pid[a]), int(nid[a]))
        s0 = first(ev, "ball_snap")
        r = dict(game_id=k3[0], play_id=k3[1], nfl_id=k3[2], n_frames=b - a, has_snap=s0 >= 0,
                 max_s=float(np.nanmax(s)) if len(s) else np.nan)
        c = first(ev, "pass_outcome_caught")
        r["s_caught"] = s[c] if c >= 0 else np.nan
        p = first(ev, "pass_arrived")
        r["s_arrived"] = s[p] if p >= 0 else np.nan
        if s0 >= 0:
            after = np.flatnonzero(s[s0:s0 + GETOFF_CAP + 1] >= GETOFF_SPEED)
            r["ol_getoff"] = (after[0] if len(after) else GETOFF_CAP) / 10
            j = min(b - a - 1, s0 + 10)
            r["dx10"], r["dy10"] = X[a + j] - X[a + s0], Y[a + j] - Y[a + s0]
        pp_rows.append(r)
        if s0 < 0:
            continue
        t = cut_t.get(k3)
        if t is not None:
            k = s0 + int(round(t * 10))
            sh = shoulders(O[a:b], D[a:b], k)
            if sh is not None:
                sh.update(game_id=k3[0], play_id=k3[1], nfl_id=k3[2])
                sh_rows.append(sh)
        if k3 in path_keys:
            hi = min(b, a + s0 + PATH_FRAMES)
            m = hi - (a + s0)
            path_rows.append(pd.DataFrame(dict(
                game_id=k3[0], play_id=k3[1], nfl_id=k3[2], frame=np.arange(m, dtype=np.int16),
                x=X[a + s0:hi].astype(np.float32), y=Y[a + s0:hi].astype(np.float32),
                s=S[a + s0:hi].astype(np.float32))))


def main():
    pl = pd.read_parquet(os.path.join(cm.OUT_DIR, "play_level.parquet"),
                         columns=["game_id", "play_id", "nfl_id", "t_after_snap"])
    cut_t = {(int(g), int(p), int(n)): float(t) for g, p, n, t in pl.itertuples(index=False)}
    pp = cm.player_play_regular_season(usecols=["game_id", "play_id", "nfl_id", "coverage_assignment"])
    plays = set(zip(pl.game_id, pl.play_id))
    man = pp[pp.coverage_assignment == "MAN"]
    man = man[[(g, p) in plays for g, p in zip(man.game_id, man.play_id)]]
    path_keys = set(cut_t) | {(int(g), int(p), int(n)) for g, p, n in zip(man.game_id, man.play_id, man.nfl_id)}
    print(f"{len(cut_t):,} Stage 3 receiver plays; {len(man):,} MAN defender rows on those plays", flush=True)
    del pp, man

    games = pd.read_csv(cm.path("games.csv"))
    reg = set(games.loc[games.season_type == "REG", "game_id"])
    cols = ["game_id", "play_id", "nfl_id", "x", "y", "s", "o", "dir", "event"]
    dtype = {"game_id": "int64", "play_id": "int64", "nfl_id": "int64", "x": "float32", "y": "float32",
             "s": "float32", "o": "float32", "dir": "float32", "event": "object"}
    pp_rows, sh_rows, path_rows = [], [], []
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
                process(ch[~last].reset_index(drop=True), cut_t, path_keys, pp_rows, sh_rows, path_rows)
        if carry is not None and len(carry):
            process(carry.reset_index(drop=True), cut_t, path_keys, pp_rows, sh_rows, path_rows)
        print(f"season {year}: {len(pp_rows):,} player-plays, {len(sh_rows):,} shoulder rows, "
              f"{len(path_rows):,} paths", flush=True)

    out = pd.DataFrame(pp_rows)
    for c in ["max_s", "s_caught", "s_arrived", "ol_getoff", "dx10", "dy10"]:
        out[c] = out[c].astype("float32")
    out.to_parquet(os.path.join(cm.OUT_DIR, "fu_player_play.parquet"), index=False)
    pd.DataFrame(sh_rows).to_parquet(os.path.join(cm.OUT_DIR, "fu_shoulders.parquet"), index=False)
    pd.concat(path_rows, ignore_index=True).to_parquet(os.path.join(cm.OUT_DIR, "fu_paths.parquet"), index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
