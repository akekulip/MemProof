"""Generate MemProof system architecture diagram as PDF.

IEEE / ACM single-column width, clean spacing, no overlapping text.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "paper" / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
})


def rounded_box(ax, x, y, w, h, text, edge_color, face_alpha=0.12,
                fontsize=8.5, weight="normal", font_color="black"):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=(*matplotlib.colors.to_rgb(edge_color), face_alpha),
        edgecolor=edge_color,
        linewidth=0.9,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=font_color)


def arrow(ax, x1, y1, x2, y2, color="black", lw=0.9, ls="-", style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                linestyle=ls, shrinkA=2, shrinkB=2))


def edge_label(ax, x, y, text, color="black", fontsize=7):
    """Place a label with a small white background for readability."""
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=color, style="italic",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.92))


def main():
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Colors
    purple = "#7C3AED"
    green = "#059669"
    orange = "#D97706"
    blue = "#2563EB"
    light_blue = "#60A5FA"
    grey = "#6B7280"

    # ------------------------------------------------------------------
    # Row layout (top to bottom):
    #   y = 8.8-9.6: External entities (Document Sources, User Query)
    #   y = 7.2-8.2: Layer 4 (Verified Retrieval) — spans right
    #   y = 5.8-6.8: Layer 1 (Provenance Ingestion) — spans left
    #   y = 4.2-5.2: Layer 2 (Embedding Integrity) — spans left
    #   y = 4.0-5.0: Layer 3 (Audit Log) — right side
    #   y = 2.4-3.2: Vector Store (center)
    #   y = 0.8-1.8: LLM Generator (right)
    # ------------------------------------------------------------------

    # External entities
    rounded_box(ax, 0.2, 8.8, 2.6, 0.9, "Document\nSources", grey, fontsize=9)
    rounded_box(ax, 11.2, 8.8, 2.6, 0.9, "User\nQuery", grey, fontsize=9)

    # Layer 1 frame (Provenance Ingestion) — left column
    ax.add_patch(FancyBboxPatch(
        (0.2, 5.4), 6.4, 2.3,
        boxstyle="round,pad=0.05",
        facecolor=(*matplotlib.colors.to_rgb(purple), 0.04),
        edgecolor=purple, linewidth=1.2, linestyle="-",
    ))
    ax.text(3.4, 7.32, "Layer 1: Provenance-Verified Ingestion",
            ha="center", fontsize=9.5, color=purple, fontweight="bold")
    rounded_box(ax, 0.5, 5.7, 1.7, 1.0, "SHA-256\nHash", purple, fontsize=8.5)
    rounded_box(ax, 2.45, 5.7, 1.9, 1.0, "Ed25519\nSign", purple, fontsize=8.5)
    rounded_box(ax, 4.6, 5.7, 1.8, 1.0, "Trust Label\n(0–4)", purple, fontsize=8.5)
    arrow(ax, 2.2, 6.2, 2.45, 6.2, purple)
    arrow(ax, 4.35, 6.2, 4.6, 6.2, purple)

    # Layer 2 frame (Embedding Integrity) — left column, below Layer 1
    ax.add_patch(FancyBboxPatch(
        (0.2, 2.9), 6.4, 2.3,
        boxstyle="round,pad=0.05",
        facecolor=(*matplotlib.colors.to_rgb(green), 0.04),
        edgecolor=green, linewidth=1.2, linestyle="-",
    ))
    ax.text(3.4, 4.82, "Layer 2: Embedding Integrity",
            ha="center", fontsize=9.5, color=green, fontweight="bold")
    rounded_box(ax, 0.5, 3.2, 1.7, 1.0, "Embed\n$E(d)$", green, fontsize=8)
    rounded_box(ax, 2.45, 3.2, 1.9, 1.0, "HMAC\nCommit", green, fontsize=8)
    rounded_box(ax, 4.6, 3.2, 1.85, 1.0, "Merkle\nInsert", green, fontsize=8)
    arrow(ax, 2.2, 3.7, 2.45, 3.7, green)
    arrow(ax, 4.35, 3.7, 4.6, 3.7, green)

    # Vector Store (center bottom)
    vstore_ellipse = Ellipse(
        (3.4, 1.5), 3.0, 1.1,
        facecolor=(*matplotlib.colors.to_rgb(blue), 0.15),
        edgecolor=blue, linewidth=1.1,
    )
    ax.add_patch(vstore_ellipse)
    ax.text(3.4, 1.5, "Vector Store", ha="center", va="center",
            fontsize=9.5, color=blue, fontweight="bold")

    # Layer 3 frame (Audit Log) — middle-right
    ax.add_patch(FancyBboxPatch(
        (7.3, 2.9), 3.2, 2.3,
        boxstyle="round,pad=0.05",
        facecolor=(*matplotlib.colors.to_rgb(orange), 0.04),
        edgecolor=orange, linewidth=1.2, linestyle="-",
    ))
    ax.text(8.9, 4.82, "Layer 3: Audit Log",
            ha="center", fontsize=9.5, color=orange, fontweight="bold")
    rounded_box(ax, 7.5, 3.2, 1.4, 1.0, "Hash\nChain", orange, fontsize=8)
    rounded_box(ax, 9.05, 3.2, 1.35, 1.0, "Drift\nDetect", orange, fontsize=8)
    arrow(ax, 8.9, 3.7, 9.05, 3.7, orange)

    # Layer 4 frame (Verified Retrieval) — right column, top
    ax.add_patch(FancyBboxPatch(
        (7.3, 5.4), 6.5, 2.3,
        boxstyle="round,pad=0.05",
        facecolor=(*matplotlib.colors.to_rgb(blue), 0.04),
        edgecolor=blue, linewidth=1.2, linestyle="-",
    ))
    ax.text(10.55, 7.32, "Layer 4: Verified Retrieval",
            ha="center", fontsize=9.5, color=blue, fontweight="bold")
    rounded_box(ax, 7.5, 5.7, 1.35, 1.0, "ANN\nSearch", blue, fontsize=8)
    rounded_box(ax, 9.0, 5.7, 1.4, 1.0, "Verify\nProofs", blue, fontsize=8)
    rounded_box(ax, 10.55, 5.7, 1.35, 1.0, "Trust\nFilter", blue, fontsize=8)
    rounded_box(ax, 12.05, 5.7, 1.65, 1.0, "Verified\nResults", blue, fontsize=8)
    arrow(ax, 8.85, 6.2, 9.0, 6.2, blue)
    arrow(ax, 10.4, 6.2, 10.55, 6.2, blue)
    arrow(ax, 11.9, 6.2, 12.05, 6.2, blue)

    # LLM Generator (right bottom)
    rounded_box(ax, 11.6, 3.2, 2.1, 1.0, "LLM\nGenerator", light_blue,
                face_alpha=0.2, fontsize=9.5, weight="bold")

    # ------------------------------------------------------------------
    # Inter-component flows (labels placed to avoid overlap)
    # ------------------------------------------------------------------

    # Document Sources -> Layer 1 (vertical, clear space)
    arrow(ax, 1.5, 8.8, 1.5, 6.75, purple, lw=1.1)
    edge_label(ax, 0.9, 7.8, "ingest", purple, fontsize=7.5)

    # Layer 1 -> Layer 2 (vertical, between frames)
    arrow(ax, 3.4, 5.4, 3.4, 5.22, green, lw=1.0)

    # Layer 2 -> Vector Store (vertical)
    arrow(ax, 3.4, 2.9, 3.4, 2.08, green, lw=1.1)
    edge_label(ax, 3.95, 2.55, "store", green, fontsize=7.5)

    # Vector Store -> Layer 4 (right side, routed around Layer 3)
    # Go horizontally right from vstore, then vertically up, then right into ANN Search
    import matplotlib.patches as mpatches
    from matplotlib.path import Path as MplPath
    verts = [
        (4.9, 1.6),    # exit right of vstore
        (7.0, 1.6),    # move right below Layer 3
        (7.0, 6.2),    # up to Layer 4 height (hugging left of Layer 3)
        (7.5, 6.2),    # arrow into ANN Search
    ]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO]
    path = MplPath(verts, codes)
    patch = mpatches.FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        color=blue,
        lw=1.0,
        linestyle="--",
        mutation_scale=12,
    )
    ax.add_patch(patch)
    edge_label(ax, 7.25, 2.4, "retrieve", blue, fontsize=7.5)

    # User Query -> Layer 4 (vertical)
    arrow(ax, 12.5, 8.8, 12.5, 6.75, blue, lw=1.1)
    edge_label(ax, 13.15, 7.8, "query", blue, fontsize=7.5)

    # Results -> LLM (vertical)
    arrow(ax, 12.87, 5.7, 12.65, 4.25, light_blue, lw=1.1)

    # Layer 2 -> Layer 3 log ops (horizontal, well below Layer 4)
    arrow(ax, 6.6, 3.7, 7.5, 3.7, orange, lw=0.9, ls="--")
    edge_label(ax, 7.05, 4.05, "log ops", orange, fontsize=7)

    plt.savefig(FIGURES_DIR / "fig_architecture.pdf", bbox_inches="tight", pad_inches=0.12)
    plt.savefig(FIGURES_DIR / "fig_architecture.png", bbox_inches="tight",
                pad_inches=0.12, dpi=200)
    plt.close()
    print(f"Saved fig_architecture.pdf to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
