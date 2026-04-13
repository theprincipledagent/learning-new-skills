"""Visualize the binomial distribution landscape using small multiples.

Each subplot shows binomial PMFs for a single n value, with line color
determined by the success probability p. Together the grid reveals how
the binomial family evolves as n grows.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.stats import binom

# --- Configuration ---
plt.style.use('seaborn-v0_8-whitegrid')
TICK_FONT_SIZE = 10

# --- Color Palette (blog theme) ---
color_palette = {
    "primary": "#0A58CA",
    "secondary": "#6A7C92",
    "accent": "#6F42C1",
    "background": "#F8F9FA",
    "text": "#212529",
}

gradient_cmap = mcolors.LinearSegmentedColormap.from_list(
    "blue_violet",
    [color_palette["primary"], color_palette["accent"]],
)

# --- Parameters ---
n_values = [5, 10, 20, 40, 70, 100]
p_values = np.linspace(0.05, 0.95, 20)
norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

# --- Plot ---
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=False)
fig.set_facecolor(color_palette["background"])

for ax, n in zip(axes.flat, n_values):
    ax.set_facecolor(color_palette["background"])

    for p in p_values:
        x = np.arange(0, n + 1)
        x_norm = x / n
        pmf = binom.pmf(x, n, p)
        color = gradient_cmap(norm(p))
        ax.plot(x_norm, pmf, color=color, alpha=0.7, linewidth=1.2)

    ax.set_title(
        f'n = {n}',
        color=color_palette["text"],
        fontsize=13,
        weight='bold',
    )
    ax.tick_params(colors=color_palette["text"], labelsize=TICK_FONT_SIZE)
    for spine in ax.spines.values():
        spine.set_edgecolor(color_palette["secondary"])

# Shared axis labels
for ax in axes[1]:
    ax.set_xlabel('k / n', color=color_palette["text"], fontsize=11)
for ax in axes[:, 0]:
    ax.set_ylabel('P(X = k)', color=color_palette["text"], fontsize=11)

# --- Colorbar ---
sm = cm.ScalarMappable(cmap=gradient_cmap, norm=norm)
sm.set_array([])
fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.3)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Success Probability (p)', color=color_palette["text"], fontsize=12)
cbar.ax.tick_params(colors=color_palette["text"], labelsize=TICK_FONT_SIZE)

fig.suptitle(
    'Binomial Distribution Landscape',
    color=color_palette["text"],
    fontsize=16,
    weight='bold',
)
plt.savefig(
    'visualizations/binomial_landscape.png',
    dpi=300,
    bbox_inches='tight',
    facecolor=fig.get_facecolor(),
)
plt.close()
print("Saved visualizations/binomial_landscape.png")
