import polars as pl

players = pl.read_csv(r"C:\Users\micha\Documents\antigravity\clever-rutherford\data\nfl-big-data-bowl-2027\players.csv", null_values=["NA"])
success = pl.read_csv(r"C:\Users\micha\Documents\antigravity\clever-rutherford\data\nfl-big-data-bowl-2027\player_career_successes.csv")

positions = ["DE", "CB", "T"]
for pos in positions:
    pos_players = players.filter(pl.col("nfl_position") == pos).join(success, on="nfl_id")
    top = pos_players.sort("career_games_started", descending=True).head(5)
    print(f"=== TOP {pos} PERFORMERS ===")
    for r in top.iter_rows(named=True):
        name = r["display_name"]
        dy = r["draft_year"]
        pk = r["draft_overall_pick"]
        st = r["career_games_started"]
        pb = r["pro_bowl_original_ballot"]
        ap = r["ap_all_pro_1st_team"]
        print(f"{name:20} | Year: {dy} Pick: {str(pk):3} | Starts: {st:2} | ProBowl: {pb} | AllPro: {ap}")
    print()
