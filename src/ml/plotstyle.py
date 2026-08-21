"""
Shared matplotlib style, so figures from different scripts read as one report.
Call apply() once, after `matplotlib.use("Agg")` and before creating any figure.
"""

import matplotlib.pyplot as plt

# Reused across figures so "predicted" is always the same blue and
# "actual/baseline" the same red.
PRIMARY = "#4C72B0"
SECONDARY = "#C44E52"
TERTIARY = "#55A868"
QUATERNARY = "#8172B2"
PALETTE = [PRIMARY, SECONDARY, TERTIARY, QUATERNARY, "#CCB974", "#64B5CD"]


def apply():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.fontsize": 9,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })
