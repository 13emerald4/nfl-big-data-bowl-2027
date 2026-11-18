import polars as pl
import plotly.graph_objects as go
import numpy as np

# Load combine tracking
comb = pl.read_csv(r"C:\Users\micha\Documents\antigravity\clever-rutherford\data\nfl-big-data-bowl-2027\combine_tracking.csv")

# Filter to Puka Nacua and Quentin Johnston on Comeback Route
puka = comb.filter((pl.col('nfl_id') == 56042) & (pl.col('drill_name') == 'COMEBACK_ROUTE_RIGHT') & (pl.col('entity_type') == 'PLAYER')).sort('time')
qj = comb.filter((pl.col('nfl_id') == 55885) & (pl.col('drill_name') == 'COMEBACK_ROUTE_RIGHT') & (pl.col('entity_type') == 'PLAYER')).sort('time')

# Normalize time to frames (0.1s increments)
num_frames = max(len(puka), len(qj))

# Create base figure
fig = go.Figure()

# Add field background layout shapes
shapes = [
    # Grass background
    dict(type="rect", x0=20, x1=45, y0=-1, y1=15, fillcolor="#1b4332", layer="below", line=dict(width=0)),
    # Sideline
    dict(type="line", x0=20, x1=45, y0=0, y1=0, line=dict(color="white", width=3)),
]

# Add yard lines
for x in range(20, 46, 5):
    lw = 2 if x % 10 == 0 else 1
    shapes.append(dict(type="line", x0=x, x1=x, y0=0, y1=15, line=dict(color="rgba(255, 255, 255, 0.4)", width=lw)))

# Trace 0: Puka Nacua full path (faint trail)
fig.add_trace(go.Scatter(
    x=puka['x'].to_list(),
    y=puka['y'].to_list(),
    mode='lines',
    line=dict(color='rgba(0, 245, 212, 0.4)', width=2, dash='dot'),
    name='Puka Nacua Path'
))

# Trace 1: Quentin Johnston full path (faint trail)
fig.add_trace(go.Scatter(
    x=qj['x'].to_list(),
    y=qj['y'].to_list(),
    mode='lines',
    line=dict(color='rgba(255, 190, 11, 0.4)', width=2, dash='dot'),
    name='Quentin Johnston Path'
))

# Trace 2: Puka Nacua moving marker
puka_s_kmh = (puka['s'] * 3.29184).to_list()
fig.add_trace(go.Scatter(
    x=[puka['x'][0]],
    y=[puka['y'][0]],
    mode='markers+text',
    marker=dict(size=14, color='#00f5d4', line=dict(color='white', width=2)),
    text=['Puka Nacua (Pick 177)'],
    textposition='top right',
    textfont=dict(color='white', size=11, family='Arial Black'),
    name='Puka Nacua'
))

# Trace 3: Quentin Johnston moving marker
qj_s_kmh = (qj['s'] * 3.29184).to_list()
fig.add_trace(go.Scatter(
    x=[qj['x'][0]],
    y=[qj['y'][0]],
    mode='markers+text',
    marker=dict(size=14, color='#ffbe0b', line=dict(color='white', width=2)),
    text=['Quentin Johnston (Pick 21)'],
    textposition='bottom right',
    textfont=dict(color='white', size=11, family='Arial Black'),
    name='Quentin Johnston'
))

# Generate animation frames
frames = []
for k in range(num_frames):
    # Puka coords
    pk_idx = min(k, len(puka) - 1)
    pk_x = puka['x'][pk_idx]
    pk_y = puka['y'][pk_idx]
    pk_spd = puka_s_kmh[pk_idx]
    pk_txt = f"Puka: {pk_spd:.1f} km/h"

    # QJ coords
    qj_idx = min(k, len(qj) - 1)
    qj_x = qj['x'][qj_idx]
    qj_y = qj['y'][qj_idx]
    qj_spd = qj_s_kmh[qj_idx]
    qj_txt = f"QJ: {qj_spd:.1f} km/h"

    frame_data = [
        # Trails remain unchanged
        go.Scatter(x=puka['x'].to_list(), y=puka['y'].to_list()),
        go.Scatter(x=qj['x'].to_list(), y=qj['y'].to_list()),
        # Moving points
        go.Scatter(x=[pk_x], y=[pk_y], text=[pk_txt]),
        go.Scatter(x=[qj_x], y=[qj_y], text=[qj_txt])
    ]
    frames.append(go.Frame(data=frame_data, name=f"frame_{k}"))

fig.frames = frames

# Animation slider and play/pause controls
fig.update_layout(
    title=dict(
        text="<b>Combine Route Kinematics Playback: Puka Nacua vs. Quentin Johnston</b><br><sup>Comeback Route Drill (10 Hz Tracking Playback)</sup>",
        font=dict(size=16, color="white")
    ),
    paper_bgcolor="#0b0f19",
    plot_bgcolor="#1b4332",
    shapes=shapes,
    xaxis=dict(
        title="Field Length (Yards)",
        range=[20, 45],
        color="white",
        gridcolor="rgba(255,255,255,0.15)",
        zeroline=False
    ),
    yaxis=dict(
        title="Field Width (Yards)",
        range=[-1, 14],
        scaleanchor="x",
        scaleratio=1,
        color="white",
        gridcolor="rgba(255,255,255,0.15)",
        zeroline=False
    ),
    legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.5)"),
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        x=0.05,
        y=1.15,
        buttons=[
            dict(
                label="▶ Play",
                method="animate",
                args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True, mode="immediate")]
            ),
            dict(
                label="⏸ Pause",
                method="animate",
                args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]
            )
        ]
    )],
    sliders=[dict(
        steps=[dict(method='animate', args=[[f.name], dict(mode='immediate', frame=dict(duration=100, redraw=True))], label=f"{i*0.1:.1f}s") for i, f in enumerate(frames)],
        transition=dict(duration=0),
        x=0.1,
        len=0.85,
        y=-0.15,
        font=dict(color="white")
    )]
)

output_html = r"C:\Users\micha\Documents\antigravity\clever-rutherford\route_playback_interactive.html"
fig.write_html(output_html)
print(f"Saved interactive animation to: {output_html}")
