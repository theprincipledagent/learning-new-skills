"""Visualize the beta distribution landscape using small multiples.

Each subplot shows beta PDFs for a fixed concentration (alpha + beta = n),
with line color determined by the mean (alpha / n). Together the grid
reveals how the beta family sharpens as concentration grows.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.stats import beta

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
# Concentration = alpha + beta. Higher concentration = tighter distribution.
concentrations = [2, 5, 10, 20, 50, 100]
# Sweep the mean = alpha / concentration
mean_values = np.linspace(0.1, 0.9, 20)
norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
x = np.linspace(0.001, 0.999, 500)

# --- Plot ---
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
fig.set_facecolor(color_palette["background"])

for ax, n in zip(axes.flat, concentrations):
    ax.set_facecolor(color_palette["background"])

    for mu in mean_values:
        a = mu * n
        b = (1 - mu) * n
        pdf = beta.pdf(x, a, b)
        color = gradient_cmap(norm(mu))
        ax.plot(x, pdf, color=color, alpha=0.7, linewidth=1.2)

    ax.set_title(
        f'$\\alpha + \\beta$ = {n}',
        color=color_palette["text"],
        fontsize=13,
        weight='bold',
    )
    ax.tick_params(colors=color_palette["text"], labelsize=TICK_FONT_SIZE)
    for spine in ax.spines.values():
        spine.set_edgecolor(color_palette["secondary"])

# Shared axis labels
for ax in axes[1]:
    ax.set_xlabel('x', color=color_palette["text"], fontsize=11)
for ax in axes[:, 0]:
    ax.set_ylabel('f(x)', color=color_palette["text"], fontsize=11)

# --- Colorbar ---
sm = cm.ScalarMappable(cmap=gradient_cmap, norm=norm)
sm.set_array([])
fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.3)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label(
    'Mean ($\\alpha / (\\alpha + \\beta)$)',
    color=color_palette["text"],
    fontsize=12,
)
cbar.ax.tick_params(colors=color_palette["text"], labelsize=TICK_FONT_SIZE)

fig.suptitle(
    'Beta Distribution Landscape',
    color=color_palette["text"],
    fontsize=16,
    weight='bold',
)

plt.savefig(
    'visualizations/beta_landscape.png',
    dpi=300,
    bbox_inches='tight',
    facecolor=fig.get_facecolor(),
)
plt.close()
print("Saved visualizations/beta_landscape.png")
