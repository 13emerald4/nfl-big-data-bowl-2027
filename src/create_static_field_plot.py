import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import polars as pl

def draw_nfl_field_section(ax, xmin=20, xmax=45, ymin=0, ymax=15):
    """Draws a clean, publication-ready slice of an NFL football field."""
    # Field background
    ax.set_facecolor('#2d6a4f') # Deep turf green
    
    # Draw yard lines every 5 yards
    for x in range(int(xmin // 5) * 5, int(xmax) + 6, 5):
        if xmin <= x <= xmax:
            lw = 1.8 if x % 10 == 0 else 0.9
            color = 'white' if x % 10 == 0 else '#e0e0e0'
            alpha = 0.85 if x % 10 == 0 else 0.5
            ax.axvline(x, ymin=0, ymax=1, color=color, lw=lw, alpha=alpha, linestyle='-')
            if x % 10 == 0:
                ax.text(x, ymax - 1.2, f"{x}", color='white', fontsize=11, 
                        fontweight='bold', ha='center', va='center', alpha=0.9)
                ax.text(x, ymin + 1.2, f"{x}", color='white', fontsize=11, 
                        fontweight='bold', ha='center', va='center', alpha=0.9)
                
    # Draw yard markers (every 1 yard along sideline)
    for x in range(int(xmin), int(xmax) + 1):
        if x % 5 != 0:
            ax.plot([x, x], [ymin, ymin + 0.6], color='white', lw=0.8, alpha=0.6)
            ax.plot([x, x], [ymax - 0.6, ymax], color='white', lw=0.8, alpha=0.6)

    # Sideline boundary
    ax.axhline(0, color='white', lw=3, alpha=0.95)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel('Field Length (X Yards)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Field Width (Y Yards)', fontsize=12, fontweight='bold', labelpad=8)

# Load data
comb = pl.read_csv(r"C:\Users\micha\Documents\antigravity\clever-rutherford\data\nfl-big-data-bowl-2027\combine_tracking.csv")

puka = comb.filter((pl.col('nfl_id') == 56042) & (pl.col('drill_name') == 'COMEBACK_ROUTE_RIGHT') & (pl.col('entity_type') == 'PLAYER')).sort('time')
qj = comb.filter((pl.col('nfl_id') == 55885) & (pl.col('drill_name') == 'COMEBACK_ROUTE_RIGHT') & (pl.col('entity_type') == 'PLAYER')).sort('time')

fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 1]})

# --- Subplot 1: Puka Nacua Trajectory ---
ax1 = axes[0]
draw_nfl_field_section(ax1, xmin=22, xmax=44, ymin=-0.5, ymax=12)

# Scatter with colormap for speed (converted to km/h)
s_kmh_puka = puka['s'].to_numpy() * 3.29184
sc1 = ax1.scatter(puka['x'], puka['y'], c=s_kmh_puka, cmap='plasma', s=65, zorder=4, edgecolor='black', lw=0.5)
ax1.plot(puka['x'], puka['y'], color='white', lw=1.5, alpha=0.7, linestyle='--', zorder=3)

# Annotate start, stem, break point, and catch
ax1.annotate('START (Release)', xy=(puka['x'][0], puka['y'][0]), xytext=(puka['x'][0]-1.5, puka['y'][0]+2.0),
             arrowprops=dict(arrowstyle="->", color="white", lw=1.2), color="white", fontweight="bold", fontsize=10)

break_idx_puka = int(np.argmax(puka['x'].to_numpy()))
ax1.annotate(f"BREAK POINT (18 yd stem)\nPeak Braking: -5.77 yd/s²\nSpeed: {s_kmh_puka[break_idx_puka]:.1f} km/h", 
             xy=(float(puka['x'][break_idx_puka]), float(puka['y'][break_idx_puka])), 
             xytext=(float(puka['x'][break_idx_puka])-7.5, float(puka['y'][break_idx_puka])-3.5),
             arrowprops=dict(arrowstyle="->", color="cyan", lw=1.8), 
             color="cyan", fontweight="bold", fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#1b263b', edgecolor='cyan', alpha=0.85))

ax1.set_title("Puka Nacua (Pick 177, All-Pro) — Combine Comeback Route Kinematics", fontsize=13, fontweight='bold', pad=10)

# --- Subplot 2: Quentin Johnston Trajectory ---
ax2 = axes[1]
draw_nfl_field_section(ax2, xmin=22, xmax=44, ymin=-0.5, ymax=12)

s_kmh_qj = qj['s'].to_numpy() * 3.29184
sc2 = ax2.scatter(qj['x'], qj['y'], c=s_kmh_qj, cmap='plasma', s=65, zorder=4, edgecolor='black', lw=0.5)
ax2.plot(qj['x'], qj['y'], color='white', lw=1.5, alpha=0.7, linestyle='--', zorder=3)

ax2.annotate('START (Release)', xy=(float(qj['x'][0]), float(qj['y'][0])), xytext=(float(qj['x'][0])-1.5, float(qj['y'][0])+2.0),
             arrowprops=dict(arrowstyle="->", color="white", lw=1.2), color="white", fontweight="bold", fontsize=10)

break_idx_qj = int(np.argmax(qj['x'].to_numpy()))
ax2.annotate(f"BREAK POINT (18 yd stem)\nPeak Braking: -5.23 yd/s²\nSpeed: {s_kmh_qj[break_idx_qj]:.1f} km/h", 
             xy=(float(qj['x'][break_idx_qj]), float(qj['y'][break_idx_qj])), 
             xytext=(float(qj['x'][break_idx_qj])-7.5, float(qj['y'][break_idx_qj])-3.5),
             arrowprops=dict(arrowstyle="->", color="#ffbe0b", lw=1.8), 
             color="#ffbe0b", fontweight="bold", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#1b263b', edgecolor='#ffbe0b', alpha=0.85))

ax2.set_title("Quentin Johnston (Pick 21, 1st Round) — Combine Comeback Route Kinematics", fontsize=13, fontweight='bold', pad=10)

# Colorbar
cbar = fig.colorbar(sc1, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.046, pad=0.15)
cbar.set_label('Instantaneous Movement Speed (km/h)', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(r"C:\Users\micha\Documents\antigravity\clever-rutherford\route_kinematics_field_plot.png", dpi=300)
print("Saved route_kinematics_field_plot.png successfully!")
