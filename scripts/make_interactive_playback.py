"""
make_interactive_playback.py
Generates the interactive Plotly HTML widgets:
1. tier_comparison_pdf_cdf.html
2. speed_pdf_cdf_interactive.html
3. route_playback_interactive.html
"""
import os
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import numpy as np

import common as cm

BASE_DIR = cm.REPO_DIR
DATA_DIR = cm.DATA_DIR

def make_all_interactive():
    print("Loading data for interactive widgets...")
    comb = pl.read_csv(os.path.join(DATA_DIR, "combine_tracking.csv"))
    players = pl.read_csv(os.path.join(DATA_DIR, "players.csv"), null_values=["NA"])
    success = pl.read_csv(os.path.join(DATA_DIR, "player_career_successes.csv"))

    # -------------------------------------------------------------
    # 1. speed_pdf_cdf_interactive.html
    # -------------------------------------------------------------
    print("Generating speed_pdf_cdf_interactive.html...")
    players_to_plot = {
        56042: ("Puka Nacua (Pick 177, All-Pro)", "#ef476f"),
        55884: ("Jaxon Smith-Njigba (Pick 20, All-Pro)", "#118ab2"),
        55885: ("Quentin Johnston (Pick 21, 1st Rd)", "#ffd166"),
        57144: ("Brian Thomas Jr. (Pick 23, 1st Rd)", "#06d6a0")
    }

    fig1 = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "<b>Speed Probability Density Function (PDF)</b>",
            "<b>Speed Cumulative Distribution Function (CDF)</b>"
        )
    )

    x_grid = np.linspace(0, 40, 300)
    for nfl_id, (label, color) in players_to_plot.items():
        s = comb.filter((pl.col("nfl_id") == nfl_id) & (pl.col("entity_type") == "PLAYER")
                        & pl.col("drill_name").is_in(list(cm.WR_CUT_DRILLS)))["s"].to_numpy()
        s_kmh = s * 3.29184
        kde = stats.gaussian_kde(s_kmh)
        pdf_vals = kde(x_grid)
        fig1.add_trace(
            go.Scatter(x=x_grid, y=pdf_vals, mode="lines", name=label, line=dict(color=color, width=3)),
            row=1, col=1
        )
        s_sorted = np.sort(s_kmh)
        cdf_vals = np.arange(1, len(s_sorted) + 1) / len(s_sorted)
        fig1.add_trace(
            go.Scatter(x=s_sorted, y=cdf_vals, mode="lines", name=label, line=dict(color=color, width=3), showlegend=False),
            row=1, col=2
        )


    fig1.update_layout(
        title=dict(
            text="<b>Cutting-drill speed distribution (curl, comeback, speed out, slant)</b><br><sup>Smith-Njigba ran only the slant and speed out; the others ran all four</sup>",
            font=dict(size=17, color="white")
        ),
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#161b26",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5, font=dict(size=11))
    )
    fig1.update_xaxes(title_text="Instantaneous Speed (km/h)", gridcolor="rgba(255,255,255,0.1)", row=1, col=1)
    fig1.update_xaxes(title_text="Instantaneous Speed (km/h)", gridcolor="rgba(255,255,255,0.1)", row=1, col=2)
    fig1.update_yaxes(title_text="Probability Density", gridcolor="rgba(255,255,255,0.1)", row=1, col=1)
    fig1.update_yaxes(title_text="Cumulative Probability F(s)", gridcolor="rgba(255,255,255,0.1)", row=1, col=2)

    p1_path = os.path.join(BASE_DIR, "speed_pdf_cdf_interactive.html")
    fig1.write_html(p1_path)
    print(f"Saved: {p1_path}")

    # -------------------------------------------------------------
    # 2. tier_comparison_pdf_cdf.html
    # -------------------------------------------------------------
    print("Generating tier_comparison_pdf_cdf.html...")
    wr_players = players.filter(pl.col("nfl_position") == "WR").join(success, on="nfl_id")
    wr_cohort = wr_players.with_columns([
        pl.when(pl.col("career_games_started") >= 20).then(pl.lit("top"))
          .when(pl.col("career_games_started") >= 3).then(pl.lit("middle"))
          .otherwise(pl.lit("fringe"))
          .alias("tier_key")
    ])
    feats = pl.read_parquet(os.path.join(cm.OUT_DIR, "features_wr.parquet"))
    wr_cohort = wr_cohort.join(feats.select(["nfl_id", "s_cut", "draft_overall_pick"]), on="nfl_id")
    summary = {k: wr_cohort.filter(pl.col("tier_key") == k) for k in ("top", "middle", "fringe")}
    label = {
        "top": f"Top (20+ starts, N={summary['top'].height})",
        "middle": f"Middle (3-19 starts, N={summary['middle'].height})",
        "fringe": f"Fringe (0-2 starts, N={summary['fringe'].height})",
    }
    wr_cohort = wr_cohort.with_columns(pl.col("tier_key").replace_strict(label).alias("performance_tier"))
    wr_cuts = comb.filter((pl.col("drill_name").is_in(list(cm.WR_CUT_DRILLS))) & (pl.col("entity_type") == "PLAYER"))
    wr_cuts = wr_cuts.join(wr_cohort, on="nfl_id")

    fig2 = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "<b>Route Cut Speed Density (PDF): Top vs. Avg vs. Poor</b>",
            "<b>Cumulative Route Cut Distribution (CDF)</b>"
        )
    )

    tiers = [
        (label["top"], "#06d6a0", 3.2),
        (label["middle"], "#ffd166", 2.5),
        (label["fringe"], "#ef476f", 2.5)
    ]
    x_grid = np.linspace(0, 35, 250)

    for tier_name, color, lw in tiers:
        speeds = (wr_cuts.filter(pl.col("performance_tier") == tier_name)["s"] * 3.29184).to_numpy()
        kde = stats.gaussian_kde(speeds)
        pdf_vals = kde(x_grid)
        fig2.add_trace(
            go.Scatter(x=x_grid, y=pdf_vals, mode="lines", name=tier_name, line=dict(color=color, width=lw)),
            row=1, col=1
        )
        s_sorted = np.sort(speeds)
        cdf_vals = np.arange(1, len(s_sorted) + 1) / len(s_sorted)
        fig2.add_trace(
            go.Scatter(x=s_sorted, y=cdf_vals, mode="lines", name=tier_name, line=dict(color=color, width=lw), showlegend=False),
            row=1, col=2
        )

    top_s, fringe_s = summary["top"]["s_cut"].to_numpy(), summary["fringe"]["s_cut"].to_numpy()
    t_stat, p_val = stats.ttest_ind(top_s, fringe_s, equal_var=False)
    top_pick = summary["top"]["draft_overall_pick"].fill_null(262).mean()
    fringe_pick = summary["fringe"]["draft_overall_pick"].fill_null(262).mean()
    fig2.add_annotation(
        text=(f"<b>Raw gap: top tier {top_s.mean() - fringe_s.mean():+.2f} km/h faster in cuts</b><br>"
              f"Welch t={t_stat:.2f}, p={p_val:.5f}<br>"
              f"But top tier was drafted far earlier (avg pick {top_pick:.0f} vs {fringe_pick:.0f}, UDFA=262);<br>"
              f"the gap no longer predicts starts once draft pick is controlled"),
        xref="x1", yref="y1",
        x=18, y=0.075,
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#06d6a0",
        ax=40, ay=-40,
        font=dict(color="white", size=11),
        bgcolor="#1b263b", bordercolor="#06d6a0", borderwidth=1.5,
        row=1, col=1
    )

    fig2.update_layout(
        title=dict(
            text=f"<b>Wide receiver cutting-drill speed by career tier</b><br><sup>{wr_cohort.height} prospects, all frames from curl, comeback, speed out and slant drills</sup>",
            font=dict(size=16, color="white")
        ),
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#161b26",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=11))
    )
    fig2.update_xaxes(title_text="Cut Speed (km/h)", gridcolor="rgba(255,255,255,0.1)", row=1, col=1)
    fig2.update_xaxes(title_text="Cut Speed (km/h)", gridcolor="rgba(255,255,255,0.1)", row=1, col=2)
    fig2.update_yaxes(title_text="Probability Density", gridcolor="rgba(255,255,255,0.1)", row=1, col=1)
    fig2.update_yaxes(title_text="Cumulative Probability F(s)", gridcolor="rgba(255,255,255,0.1)", row=1, col=2)

    p2_path = os.path.join(BASE_DIR, "tier_comparison_pdf_cdf.html")
    fig2.write_html(p2_path)
    print(f"Saved: {p2_path}")

    # -------------------------------------------------------------
    # 3. route_playback_interactive.html
    # -------------------------------------------------------------
    print("Generating route_playback_interactive.html...")
    puka = comb.filter((pl.col("nfl_id") == 56042) & (pl.col("drill_name") == "COMEBACK_ROUTE_RIGHT") & (pl.col("entity_type") == "PLAYER")).sort("time")
    qj = comb.filter((pl.col("nfl_id") == 55885) & (pl.col("drill_name") == "COMEBACK_ROUTE_RIGHT") & (pl.col("entity_type") == "PLAYER")).sort("time")

    num_frames = max(len(puka), len(qj))
    fig3 = go.Figure()

    shapes = [
        dict(type="rect", x0=20, x1=45, y0=-1, y1=15, fillcolor="#1b4332", layer="below", line=dict(width=0)),
        dict(type="line", x0=20, x1=45, y0=0, y1=0, line=dict(color="white", width=3)),
    ]
    for x in range(20, 46, 5):
        lw = 2 if x % 10 == 0 else 1
        shapes.append(dict(type="line", x0=x, x1=x, y0=0, y1=15, line=dict(color="rgba(255, 255, 255, 0.4)", width=lw)))

    fig3.add_trace(go.Scatter(x=puka["x"].to_list(), y=puka["y"].to_list(), mode="lines",
                              line=dict(color="rgba(0, 245, 212, 0.4)", width=2, dash="dot"), name="Puka Nacua Path"))
    fig3.add_trace(go.Scatter(x=qj["x"].to_list(), y=qj["y"].to_list(), mode="lines",
                              line=dict(color="rgba(255, 190, 11, 0.4)", width=2, dash="dot"), name="Quentin Johnston Path"))

    puka_s_kmh = (puka["s"] * 3.29184).to_list()
    fig3.add_trace(go.Scatter(x=[puka["x"][0]], y=[puka["y"][0]], mode="markers+text",
                              marker=dict(size=14, color="#00f5d4", line=dict(color="white", width=2)),
                              text=["Puka Nacua (Pick 177)"], textposition="top right",
                              textfont=dict(color="white", size=11, family="Arial Black"), name="Puka Nacua"))

    qj_s_kmh = (qj["s"] * 3.29184).to_list()
    fig3.add_trace(go.Scatter(x=[qj["x"][0]], y=[qj["y"][0]], mode="markers+text",
                              marker=dict(size=14, color="#ffbe0b", line=dict(color="white", width=2)),
                              text=["Quentin Johnston (Pick 21)"], textposition="bottom right",
                              textfont=dict(color="white", size=11, family="Arial Black"), name="Quentin Johnston"))

    frames = []
    for k in range(num_frames):
        pk_idx = min(k, len(puka) - 1)
        pk_x = puka["x"][pk_idx]
        pk_y = puka["y"][pk_idx]
        pk_spd = puka_s_kmh[pk_idx]
        pk_txt = f"Puka: {pk_spd:.1f} km/h"

        qj_idx = min(k, len(qj) - 1)
        qj_x = qj["x"][qj_idx]
        qj_y = qj["y"][qj_idx]
        qj_spd = qj_s_kmh[qj_idx]
        qj_txt = f"QJ: {qj_spd:.1f} km/h"

        frame_data = [
            go.Scatter(x=puka["x"].to_list(), y=puka["y"].to_list()),
            go.Scatter(x=qj["x"].to_list(), y=qj["y"].to_list()),
            go.Scatter(x=[pk_x], y=[pk_y], text=[pk_txt]),
            go.Scatter(x=[qj_x], y=[qj_y], text=[qj_txt])
        ]
        frames.append(go.Frame(data=frame_data, name=f"frame_{k}"))

    fig3.frames = frames

    sliders = [dict(
        steps=[dict(method="animate", args=[[f"frame_{k}"], dict(mode="immediate", frame=dict(duration=100, redraw=True), transition=dict(duration=0))],
                    label=f"{k/10:.1f}s") for k in range(num_frames)],
        transition=dict(duration=0),
        x=0.1, y=0, currentvalue=dict(font=dict(size=12, color="white"), prefix="Drill Time: ", visible=True, xanchor="right"),
        len=0.85, pad=dict(b=10, t=10), font=dict(color="white")
    )]

    fig3.update_layout(
        title="<b>Combine Comeback Route: 10 Hz Optical Tracking Comparison</b><br><sup>Puka Nacua (Pick 177, All-Pro) vs. Quentin Johnston (Pick 21, 1st Rd)</sup>",
        title_font=dict(size=16, color="white"),
        shapes=shapes,
        xaxis=dict(range=[20, 45], title="Field Length (Yards)", showgrid=False, zeroline=False, color="white"),
        yaxis=dict(range=[-1, 14], title="Field Width (Yards)", showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, color="white"),
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#1b4332",
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.02, y=-0.15,
            buttons=[
                dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True, mode="immediate")]),
                dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
            ]
        )],
        sliders=sliders,
        legend=dict(font=dict(color="white"), orientation="h", y=1.05, x=0.2)
    )

    p3_path = os.path.join(BASE_DIR, "route_playback_interactive.html")
    fig3.write_html(p3_path)
    print(f"Saved: {p3_path}")
    print("All interactive HTML widgets generated successfully!")

if __name__ == "__main__":
    make_all_interactive()
