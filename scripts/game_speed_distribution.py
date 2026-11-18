"""
game_speed_distribution.py
Regular-season speed distribution from 10 Hz game tracking (all three seasons).
Streams the CSVs in chunks, so the 2 GB of tracking never has to sit in memory.
If the game_tracking CSVs are not extracted, set NFL_BDB_ZIP to the competition
zip and they are read from it directly.

Writes results/game_speed_distribution.txt.
"""
import os

import numpy as np
import pandas as pd

import common as cm

EDGES = np.arange(0, 15.0001, 0.005)          # yd/s histogram bins
LO, HI = 10 / cm.YDS_TO_KMH, 26 / cm.YDS_TO_KMH  # 10 and 26 km/h in yd/s


def main():
    games = pd.read_csv(cm.path("games.csv"))
    reg = set(games.loc[games.season_type == "REG", "game_id"])
    hist = np.zeros(len(EDGES) - 1)
    over = 0
    maxes = []
    for year in (2023, 2024, 2025):
        with cm.open_game_tracking(year) as f:
            reader = pd.read_csv(f, usecols=["game_id", "play_id", "nfl_id", "s"], chunksize=1_000_000,
                                 dtype={"game_id": "int64", "play_id": "int32", "nfl_id": "int32", "s": "float32"})
            for ch in reader:
                ch = ch[ch.game_id.isin(reg)]
                s = ch.s.to_numpy()
                hist += np.histogram(s, EDGES)[0]
                over += int((s >= EDGES[-1]).sum())
                maxes.append(ch.groupby(["game_id", "play_id", "nfl_id"], sort=False).s.max())
                if len(maxes) >= 20:
                    maxes = [pd.concat(maxes).groupby(level=[0, 1, 2], sort=False).max()]
        print(f"season {year} done", flush=True)
    n = hist.sum() + over
    cum = np.cumsum(hist)
    q = lambda p: EDGES[np.searchsorted(cum, p * n) + 1]
    centers = EDGES[:-1]
    below = hist[centers < LO].sum() / n
    band = hist[(centers >= LO) & (centers < HI)].sum() / n
    above = (hist[centers >= HI].sum() + over) / n
    mx = pd.concat(maxes).groupby(level=[0, 1, 2]).max()
    lines = [
        f"Regular-season frames: {int(n):,}",
        f"Median speed: {q(.5):.2f} yd/s ({q(.5)*cm.YDS_TO_KMH:.1f} km/h)",
        f"95th percentile: {q(.95):.2f} yd/s ({q(.95)*cm.YDS_TO_KMH:.1f} km/h)",
        f"Share of frames below 10 km/h: {below:.3f}",
        f"Share of frames in 10-26 km/h: {band:.3f}",
        f"Share of frames above 26 km/h: {above:.4f}",
        f"Player-snaps with tracking: {len(mx):,}",
        f"Share of player-snaps whose top speed exceeds 26 km/h: {(mx > HI).mean():.3f}",
    ]
    os.makedirs(cm.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(cm.RESULTS_DIR, "game_speed_distribution.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
