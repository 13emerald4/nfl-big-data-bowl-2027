"""
common.py
Shared paths, loaders and feature definitions for every analysis script.

Data location: the folder holding the competition CSVs. Defaults to
../data/nfl-big-data-bowl-2027 (a sibling of this repository); override with
the NFL_BDB_DATA environment variable. Game tracking (2 GB) can also be read
straight from the downloaded zip, see open_game_tracking().
"""
import os
import zipfile

import numpy as np
import pandas as pd
import polars as pl

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("NFL_BDB_DATA", os.path.join(REPO_DIR, "..", "data", "nfl-big-data-bowl-2027"))
OUT_DIR = os.path.join(REPO_DIR, "data")          # derived features (git-ignored)
RESULTS_DIR = os.path.join(REPO_DIR, "results")   # model output tables (committed)

YDS_TO_KMH = 3.29184   # 1 yd/s = 0.9144 m/s = 3.29184 km/h
LB_TO_KG = 0.45359237
UDFA_PICK = 262        # undrafted free agents sit after the last pick (259 in 2023-2025)
LAST_SEASON = 2025     # career outcomes run through the 2025 season

# Cutting drills used for receivers. SPEED_OUT_LEFT is the same drill as
# SPEED_OUT_ROUTE_LEFT under a different name in one draft class.
WR_CUT_DRILLS = {
    "CURL_ROUTE_RIGHT": "curl",
    "COMEBACK_ROUTE_RIGHT": "comeback",
    "SPEED_OUT_ROUTE_LEFT": "speed_out",
    "SPEED_OUT_LEFT": "speed_out",
    "SLANT_ROUTE_LEFT": "slant",
}
WR_POS = ["WR"]
DL_POS = ["DE", "DT", "NT", "OLB"]
DB_POS = ["CB", "SS", "FS", "DB"]
OL_POS = ["T", "G", "C"]
OL_DRILLS = ["PASS_PRO_MIRROR_DRILL", "PASS_PRO-MIRROR_DRILL", "FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE"]
DB_DRILL = "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION"
DL_DRILL = "PASS_RUSH_DRILL"


def path(name):
    return os.path.join(DATA_DIR, name)


def combine_tracking():
    """Combine tracking, PLAYER rows only, ordered within each drill attempt."""
    return (pl.read_csv(path("combine_tracking.csv"))
            .filter(pl.col("entity_type") == "PLAYER")
            .sort(["event_id", "time"]))


def player_base():
    """One row per prospect: bio, draft slot, combine results, career outcomes."""
    players = pd.read_csv(path("players.csv"))
    results = pd.read_csv(path("combine_results.csv")).drop(columns=["draft_year"])
    careers = pd.read_csv(path("player_career_successes.csv"))
    df = players.merge(results, on="nfl_id").merge(careers, on="nfl_id")
    df["udfa"] = df["draft_overall_pick"].isna().astype(int)
    df["log_pick"] = np.log(df["draft_overall_pick"].fillna(UDFA_PICK))
    df["seasons"] = LAST_SEASON - df["draft_year"] + 1
    df["starter"] = (df["career_games_started"] >= 10).astype(int)
    df["draft_day"] = np.select(
        [df["draft_round"] == 1, df["draft_round"].isin([2, 3]), df["draft_round"] >= 4],
        ["Day 1", "Day 2", "Day 3"], "UDFA")
    return df


def player_play_regular_season():
    """Per-snap game records, regular season only. 'NA' strings become missing."""
    pp = pd.read_csv(path("player_play.csv"), keep_default_na=False, na_values=["", "NA"], low_memory=False)
    games = pd.read_csv(path("games.csv"))
    reg = set(games.loc[games["season_type"] == "REG", "game_id"])
    return pp[pp["game_id"].isin(reg)].copy()


def open_game_tracking(year, zip_path=None):
    """File handle for game_tracking_<year>.csv, from DATA_DIR or from the zip."""
    name = f"game_tracking_{year}.csv"
    if os.path.exists(path(name)):
        return open(path(name), "rb")
    zip_path = zip_path or os.environ.get("NFL_BDB_ZIP")
    if not zip_path:
        raise FileNotFoundError(f"{name} not in {DATA_DIR}; set NFL_BDB_ZIP to the competition zip")
    z = zipfile.ZipFile(zip_path)
    member = next(n for n in z.namelist() if n.endswith(name))
    return z.open(member)
