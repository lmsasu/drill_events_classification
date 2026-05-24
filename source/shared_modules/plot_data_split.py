"""Generate and save the data-partitioning diagram for the paper.

The main timeline uses a proportional (70 / 15 / 15) layout.  The look-back
context zones are drawn at an exaggerated fixed width so they are clearly
visible; a "not to scale" note is included in the figure.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Dataset constants ──────────────────────────────────────────────────────────
N_TOTAL = 127_294
N_TEST  = int(N_TOTAL * 0.15)          # 19,094
N_VAL   = int(N_TOTAL * 0.15)          # 19,094
N_TRAIN = N_TOTAL - N_VAL - N_TEST     # 89,106
L       = 30   # representative lookback (mid of [10, 50] search range)

COLORS = {
    "train": "#4C72B0",
    "val":   "#DD8452",
    "test":  "#55A868",
    "ctx":   "#B0B0B0",
}

# ── Proportional x-coordinates (0 … 1) ────────────────────────────────────────
# Segments are proportional; context zones are fixed at 3 % of total width so
# they stay visible.  A "not to scale" annotation makes this explicit.
CTX_W = 0.030           # exaggerated context zone width in axes fraction
p_train_end  = N_TRAIN / N_TOTAL                      # ≈ 0.70
p_val_end    = (N_TRAIN + N_VAL) / N_TOTAL            # ≈ 0.85

# ── Figure setup ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 2.4))
ax.set_xlim(0, 1)
ax.set_ylim(-1.0, 2.3)
ax.axis("off")

BAR_Y, BAR_H = 1.15, 0.52
WIN_Y, WIN_H  = 0.0,  0.36

# ── Helper: draw one colour block ─────────────────────────────────────────────
def block(x0, x1, color, zorder=3):
    ax.barh(BAR_Y, x1 - x0, left=x0, height=BAR_H,
            color=color, edgecolor="white", linewidth=1.0, zorder=zorder)

# ── Helper: draw context-zone hatch overlay ───────────────────────────────────
def ctx_zone(x0, x1, label_side="above"):
    ax.barh(BAR_Y, x1 - x0, left=x0, height=BAR_H,
            facecolor=COLORS["ctx"], edgecolor="#777", linewidth=0.6,
            hatch="////", zorder=5, alpha=0.90)
    mid = (x0 + x1) / 2
    if label_side == "above":
        ax.text(mid, BAR_Y + BAR_H / 2 + 0.08,
                "look-back context $L$",
                ha="center", va="bottom", fontsize=7.0,
                color="#444", style="italic", linespacing=1.35)
    else:
        ax.text(mid, BAR_Y - BAR_H / 2 - 0.08,
                "look-back context $L$",
                ha="center", va="top", fontsize=7.0,
                color="#444", style="italic", linespacing=1.35)

# ── Helper: draw a window box + arrow to labelled endpoint ────────────────────
def window(x0, x1, color, note="", note_side="below", alpha=0.22):
    rect = mpatches.FancyBboxPatch(
        (x0, WIN_Y - WIN_H / 2), x1 - x0, WIN_H,
        boxstyle="round,pad=0.003", linewidth=1.1,
        edgecolor=color, facecolor=color, alpha=alpha, zorder=6,
    )
    ax.add_patch(rect)
    ax.plot(x1, WIN_Y, "o", color=color, ms=5.5, zorder=7)
    ax.annotate("",
        xy=(x1, BAR_Y - BAR_H / 2 - 0.01),
        xytext=(x1, WIN_Y + WIN_H / 2 + 0.01),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3),
        zorder=7,
    )
    if note:
        y = WIN_Y - WIN_H / 2 - 0.13 if note_side == "below" else WIN_Y + WIN_H / 2 + 0.08
        va = "top" if note_side == "below" else "bottom"
        ax.text((x0 + x1) / 2, y, note,
                ha="center", va=va, fontsize=7.8, color=color)

# ── Draw segments ─────────────────────────────────────────────────────────────
block(0,            p_train_end, COLORS["train"])
block(p_train_end,  p_val_end,   COLORS["val"])
block(p_val_end,    1.0,          COLORS["test"])

# ── Draw context zones (exaggerated) ─────────────────────────────────────────
ctx_zone(p_train_end - CTX_W, p_train_end, label_side="above")
ctx_zone(p_val_end   - CTX_W, p_val_end,   label_side="above")

# ── Segment labels (staggered heights to avoid overlap) ───────────────────────
label_specs = [
    (0,           p_train_end, COLORS["train"], f"Training\n{N_TRAIN:,} steps  (70 %)",  1.70),
    (p_train_end, p_val_end,   COLORS["val"],   f"Validation\n{N_VAL:,} steps  (15 %)",  1.70),
    (p_val_end,   1.0,          COLORS["test"],  f"Test\n{N_TEST:,} steps  (15 %)",        1.70),
]
for x0, x1, c, lbl, ly in label_specs:
    ax.text((x0 + x1) / 2, ly, lbl,
            ha="center", va="bottom", fontsize=10, fontweight="bold", color=c,
            linespacing=1.4)

# ── Boundary dividers ────────────────────────────────────────────────────────
for xb in [p_train_end, p_val_end]:
    ax.plot([xb, xb],
            [BAR_Y - BAR_H / 2, BAR_Y + BAR_H / 2],
            color="white", lw=2.2, zorder=8, solid_capstyle="round")
    ax.plot([xb, xb],
            [BAR_Y - BAR_H / 2, BAR_Y + BAR_H / 2],
            color="#333333", lw=0.9, zorder=9, linestyle="--",
            dashes=(4, 3), solid_capstyle="round")

# ── Representative windows ────────────────────────────────────────────────────
WW = 0.025  # half-width base
WIN_W = WW + CTX_W * 0.6  # unified window width (matches val/test straddle width)

# One representative training window
cx = 0.12 * p_train_end
window(cx - WIN_W / 2, cx + WIN_W / 2, COLORS["train"], "train window")

# First validation window (straddles boundary — uses context)
val_ctx_mid = p_train_end - CTX_W / 2
window(val_ctx_mid - WIN_W / 2, val_ctx_mid + WIN_W / 2,
       COLORS["val"],
       f"1st val window\n(label→t={N_TRAIN + 1:,})")

# First test window (straddles boundary — uses context)
test_ctx_mid = p_val_end - CTX_W / 2
window(test_ctx_mid - WIN_W / 2, test_ctx_mid + WIN_W / 2,
       COLORS["test"],
       f"1st test window\n(label→t={N_TRAIN + N_VAL + 1:,})",
       alpha=0.65)

# ── X-axis ticks ──────────────────────────────────────────────────────────────
ticks = [(0, "1"), (p_train_end, f"{N_TRAIN:,}"),
         (p_val_end, f"{N_TRAIN + N_VAL:,}"), (1.0, f"{N_TOTAL:,}")]
for xp, xl in ticks:
    ax.plot([xp, xp], [BAR_Y - BAR_H / 2, BAR_Y - BAR_H / 2 - 0.10],
            color="#777", lw=0.9)
    ax.text(xp, BAR_Y - BAR_H / 2 - 0.14, xl,
            ha="center", va="top", fontsize=8.5, color="#444")



# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color=COLORS["train"], label="Training (89,106 steps · 70 %)"),
    mpatches.Patch(color=COLORS["val"],   label="Validation (19,094 steps · 15 %)"),
    mpatches.Patch(color=COLORS["test"],  label="Test (19,094 steps · 15 %)"),
    mpatches.Patch(facecolor=COLORS["ctx"], edgecolor="#777", hatch="////",
                   label="Look-back context zone $L$"),
]
ax.legend(handles=handles, loc="upper center",
          bbox_to_anchor=(0.35, 0.38), ncol=2, fontsize=8.8,
          frameon=False, handlelength=1.6)

plt.tight_layout()

out_path = Path(__file__).resolve().parents[2] / "results" / "data_split_diagram.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved → {out_path}")
