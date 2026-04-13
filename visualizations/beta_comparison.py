"""Monte Carlo comparison of Beta distributions.

Samples from each distribution, computes pairwise probability that one
is greater than the other, and the expected magnitude of the difference.
Outputs two heatmap matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Configuration ---
plt.style.use('seaborn-v0_8-whitegrid')
TICK_FONT_SIZE = 11
N_SAMPLES = 1_000_000
np.random.seed(42)

# --- Color Palette (blog theme) ---
color_palette = {
    "primary": "#0A58CA",
    "secondary": "#6A7C92",
    "accent": "#6F42C1",
    "background": "#F8F9FA",
    "text": "#212529",
}

prob_cmap = mcolors.LinearSegmentedColormap.from_list(
    "white_violet",
    ["#FFFFFF", color_palette["accent"]],
)
diff_cmap = mcolors.LinearSegmentedColormap.from_list(
    "blue_white_violet",
    [color_palette["primary"], "#FFFFFF", color_palette["accent"]],
)

# --- Distributions ---
distributions = {
    "Pre-Skill\nTest":  (30, 59),
    "Post-Skill\nTest": (246, 499),
    "Pre-Skill\nTrain": (37, 52),
    "Post-Skill\nTrain": (313, 432),
}

names = list(distributions.keys())
n = len(names)

# Draw samples
samples = {}
for name, (a, b) in distributions.items():
    samples[name] = np.random.beta(a, b, size=N_SAMPLES)

# --- Compute pairwise matrices ---
prob_greater = np.zeros((n, n))
mean_diff = np.zeros((n, n))

for i, name_i in enumerate(names):
    for j, name_j in enumerate(names):
        diff = samples[name_i] - samples[name_j]
        prob_greater[i, j] = np.mean(diff > 0)
        mean_diff[i, j] = np.mean(diff)

# --- Print summary ---
print(f"Monte Carlo simulation ({N_SAMPLES:,} samples)\n")
print("Distribution means:")
for name, (a, b) in distributions.items():
    mean = a / (a + b)
    print(f"  {name.replace(chr(10), ' '):20s}: {mean:.4f}  (α={a}, β={b})")

col_w = 20
short = [name.replace("\n", " ") for name in names]

def print_matrix(title, matrix, fmt):
    print(f"\n{title}")
    print("-" * (col_w + col_w * n))
    print("".ljust(col_w) + "".join(s.rjust(col_w) for s in short))
    print("-" * (col_w + col_w * n))
    for i, name_i in enumerate(short):
        row = name_i.ljust(col_w)
        row += "".join(fmt(matrix[i, j]).rjust(col_w) for j in range(n))
        print(row)
    print()

print_matrix("P(row > col)", prob_greater, lambda v: f"{v:.4f}")
print_matrix("E[row - col]", mean_diff, lambda v: f"{v:+.4f}")
print_matrix("E[row - col] (pct pts)", mean_diff * 100, lambda v: f"{v:+.2f}pp")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.set_facecolor(color_palette["background"])

short_names = [name.replace("\n", " ") for name in names]

# --- Heatmap 1: P(row > col) ---
ax1.set_facecolor(color_palette["background"])
im1 = ax1.imshow(prob_greater, cmap=prob_cmap, vmin=0, vmax=1, aspect='equal')
ax1.set_xticks(range(n))
ax1.set_yticks(range(n))
ax1.set_xticklabels(short_names, fontsize=TICK_FONT_SIZE, color=color_palette["text"], rotation=30, ha='right')
ax1.set_yticklabels(short_names, fontsize=TICK_FONT_SIZE, color=color_palette["text"])
ax1.set_title('P(row > col)', fontsize=14, weight='bold', color=color_palette["text"])
for i in range(n):
    for j in range(n):
        val = prob_greater[i, j]
        text_color = "white" if val > 0.6 else color_palette["text"]
        ax1.text(j, i, f"{val:.3f}", ha='center', va='center', fontsize=11, color=text_color, weight='bold')
for spine in ax1.spines.values():
    spine.set_edgecolor(color_palette["secondary"])

cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
cbar1.set_label('Probability', color=color_palette["text"], fontsize=11)
cbar1.ax.tick_params(colors=color_palette["text"], labelsize=TICK_FONT_SIZE)

# --- Heatmap 2: E[row - col] ---
max_abs = np.max(np.abs(mean_diff))

ax2.set_facecolor(color_palette["background"])
im2 = ax2.imshow(mean_diff, cmap=diff_cmap, vmin=-max_abs, vmax=max_abs, aspect='equal')
ax2.set_xticks(range(n))
ax2.set_yticks(range(n))
ax2.set_xticklabels(short_names, fontsize=TICK_FONT_SIZE, color=color_palette["text"], rotation=30, ha='right')
ax2.set_yticklabels(short_names, fontsize=TICK_FONT_SIZE, color=color_palette["text"])
ax2.set_title('E[row - col]', fontsize=14, weight='bold', color=color_palette["text"])
for i in range(n):
    for j in range(n):
        val = mean_diff[i, j]
        text_color = "white" if abs(val) > max_abs * 0.5 else color_palette["text"]
        ax2.text(j, i, f"{val:+.4f}", ha='center', va='center', fontsize=11, color=text_color, weight='bold')
for spine in ax2.spines.values():
    spine.set_edgecolor(color_palette["secondary"])

cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
cbar2.set_label('Mean Difference', color=color_palette["text"], fontsize=11)
cbar2.ax.tick_params(colors=color_palette["text"], labelsize=TICK_FONT_SIZE)

fig.suptitle(
    'Pairwise Beta Distribution Comparison',
    fontsize=16, weight='bold', color=color_palette["text"],
)
plt.tight_layout()
plt.savefig(
    'visualizations/beta_comparison.png',
    dpi=300,
    bbox_inches='tight',
    facecolor=fig.get_facecolor(),
)
plt.close()
print("\nSaved visualizations/beta_comparison.png")
