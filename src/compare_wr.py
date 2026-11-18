import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# Load combine tracking
comb = pl.read_csv(r"C:\Users\micha\Documents\antigravity\clever-rutherford\data\nfl-big-data-bowl-2027\combine_tracking.csv")

target_drills = ["CURL_ROUTE_RIGHT", "COMEBACK_ROUTE_RIGHT", "SLANT_ROUTE_LEFT"]

players_to_check = {
    56042: "Puka Nacua (Pick 177, All-Pro)",
    55884: "JSN (Pick 20, All-Pro)",
    55885: "Quentin Johnston (Pick 21)",
    57144: "Brian Thomas Jr. (Pick 23)"
}

subset = comb.filter((pl.col("nfl_id").is_in(list(players_to_check.keys()))) & 
                     (pl.col("entity_type") == "PLAYER") &
                     (pl.col("drill_name").is_in(target_drills)))

res = subset.group_by(["nfl_id", "drill_name"]).agg([
    pl.col("s").mean().alias("mean_speed"),
    pl.col("s").quantile(0.95).alias("p95_speed"),
    pl.col("a").min().alias("peak_decel"),
    (pl.col("a") < -3.0).mean().alias("pct_hard_braking")
]).sort(["drill_name", "peak_decel"])

print("--- ROUTE DRILL BRAKING & SPEED COMPARISON ---")
for r in res.iter_rows(named=True):
    name = players_to_check[r["nfl_id"]]
    dname = r["drill_name"]
    pdec = round(r["peak_decel"], 2)
    p95 = round(r["p95_speed"], 2)
    print(f"{name:32} | {dname:20} | Peak Decel (a): {pdec:6.2f} yd/s^2 | P95 Speed: {p95:5.2f} yd/s")
