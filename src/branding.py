"""
Chantel Hill brand styling for matplotlib charts.

Applies the project's editorial palette (forest green / charcoal / cream),
loads Playfair Display and Jost from assets/fonts, and exposes named color
constants so chart code reads BRAND_FOREST instead of raw hex.

Visual rules baked in:
  - No rounded corners on bars (patch.linewidth = 0; sharp edges).
  - Grid lines are #E4E0D8, thin, y-axis only by default.
  - Forest green is used as a fill color on cream/white backgrounds, never
    as a chart background.
  - Top and right spines removed for an editorial look.
  - Body text in Jost, chart titles in Playfair Display.

Call `apply_style()` once at the start of any chart-producing script.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager


# ---------------------------------------------------------------------------
# Color palette (matches index.html CSS variables)
# ---------------------------------------------------------------------------

WHITE = "#FFFFFF"
OFF_WHITE = "#F7F5F1"     # cream background
PAPER = "#F0EDE7"
FOREST = "#2C4A35"        # primary brand color
FOREST_DARK = "#1E3326"
FOREST_LIGHT = "#3D6147"
CHARCOAL = "#1C1C1A"      # primary text
MID = "#737067"           # secondary text
LIGHT = "#DDD9D1"
RULE = "#E4E0D8"          # divider lines

# Semantic roles for two-series comparison charts.
# REFERENCE is the anchor (the answer key, the truth, the configured value).
# SUBJECT is the thing being scrutinized (the model claim, the measurement).
REFERENCE = CHARCOAL
SUBJECT = FOREST

# Sequence palette for multi-series charts (matplotlib prop cycle).
SERIES_PALETTE = [FOREST, CHARCOAL, FOREST_LIGHT, MID, FOREST_DARK, LIGHT]


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

_FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
_FONTS_REGISTERED = False


def _register_fonts() -> None:
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED or not _FONT_DIR.exists():
        return
    for ttf in _FONT_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(str(ttf))
    _FONTS_REGISTERED = True


def title_font(size: float = 14, italic: bool = False, weight: str = "bold"):
    """FontProperties for chart titles. Playfair Display, charcoal, bold by default."""
    return font_manager.FontProperties(
        family="Playfair Display",
        size=size,
        weight=weight,
        style="italic" if italic else "normal",
    )


def eyebrow_font(size: float = 8):
    """FontProperties for ALL CAPS eyebrow labels above the chart title."""
    return font_manager.FontProperties(family="Jost", size=size, weight="600")


def body_font(size: float = 10, weight: str = "400"):
    return font_manager.FontProperties(family="Jost", size=size, weight=weight)


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """Apply the brand style to matplotlib. Idempotent; safe to call repeatedly."""
    _register_fonts()

    mpl.rcParams.update({
        # Figure / save
        "figure.facecolor": OFF_WHITE,
        "savefig.facecolor": OFF_WHITE,
        "savefig.edgecolor": "none",
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.4,

        # Axes
        "axes.facecolor": OFF_WHITE,
        "axes.edgecolor": CHARCOAL,
        "axes.labelcolor": CHARCOAL,
        "axes.titlecolor": CHARCOAL,
        "axes.titleweight": "normal",
        "axes.titlesize": 12,
        "axes.titlepad": 18,
        "axes.labelsize": 10,
        "axes.labelpad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.8,

        # Grid
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": RULE,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,

        # Ticks (no tick marks; just labels with breathing room)
        "xtick.color": MID,
        "ytick.color": MID,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,

        # Series
        "axes.prop_cycle": mpl.cycler(color=SERIES_PALETTE),
        "lines.linewidth": 1.6,

        # Bars: sharp edges, no border
        "patch.linewidth": 0,
        "patch.edgecolor": CHARCOAL,

        # Legend
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": CHARCOAL,

        # Fonts: Jost for body, Playfair Display reachable via FontProperties
        "font.family": "sans-serif",
        "font.sans-serif": ["Jost", "Helvetica", "Arial", "DejaVu Sans"],
        "font.serif": ["Playfair Display", "Georgia", "DejaVu Serif"],
        "font.size": 10,
    })


# ---------------------------------------------------------------------------
# Editorial helpers
# ---------------------------------------------------------------------------

EYEBROW_SEP = "  "  # two regular spaces approximate letter-spacing without font glyph issues


def add_eyebrow(fig, text: str, *, x: float = 0.06, y: float = 0.965) -> None:
    """
    Place an ALL CAPS, wide-letter-spaced eyebrow label above the chart title.

    Mirrors the editorial pattern in Chantel's portfolio HTML:
    `letter-spacing: 0.16em; text-transform: uppercase; color: var(--mid)`.
    matplotlib has no native letter-spacing, so we approximate by joining
    characters with thin spaces.
    """
    spaced = EYEBROW_SEP.join(list(text.upper()))
    fig.text(x, y, spaced, fontproperties=eyebrow_font(), color=MID)


def add_titles(fig, ax, *, title: str, subtitle: str | None = None) -> None:
    """
    Two-line title pattern: Playfair Display headline in charcoal, with an
    optional Jost subtitle in mid gray underneath. Replaces ax.set_title.
    """
    ax.set_title(title, fontproperties=title_font(size=14), color=CHARCOAL, loc="left")
    if subtitle:
        ax.text(
            0.0, 1.005, subtitle,
            transform=ax.transAxes,
            fontproperties=body_font(size=9, weight="300"),
            color=MID,
            verticalalignment="bottom",
        )


def make_forest_cmap():
    """Single-hue cream-to-forest colormap for heatmaps."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "chantel_forest",
        [OFF_WHITE, LIGHT, FOREST_LIGHT, FOREST, FOREST_DARK],
    )
