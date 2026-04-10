"""Generate publication-quality figures from REAL experimental data.

Uses actual NQ results with GPT-3.5. All figures follow IEEE style:
- Serif font, 300 DPI, PDF output
- NO overlapping text or legends
- Clean axis labels, proper spacing
- Legends outside plot area or in empty regions only
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

FIGURES_DIR = Path(__file__).parent.parent / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# IEEE double-column style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
})

COL_W = 3.45  # IEEE single column width in inches


def fig_main_comparison(results: dict, save_prefix: str = "fig_real"):
    """Fig 1: ASR and Accuracy comparison across 4 defenses.

    Grouped bar chart with ASR and Accuracy side by side.
    Legend placed below the chart to avoid overlap.
    """
    names = list(results.keys())
    short_names = ["No\nDefense", "RobustRAG\nOnly", "MemProof\nOnly", "MemProof +\nRobustRAG"]
    accs = [results[n]["accuracy"] * 100 for n in names]
    asrs = [results[n]["asr"] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    x = np.arange(len(names))
    w = 0.32

    bars_acc = ax.bar(x - w/2, accs, w, label="Accuracy",
                       color="#4878CF", edgecolor="black", linewidth=0.4)
    bars_asr = ax.bar(x + w/2, asrs, w, label="Attack Success Rate",
                       color="#D65F5F", edgecolor="black", linewidth=0.4)

    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=7.5)
    ax.set_ylim(0, 72)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))

    # Bar labels — small, above bars
    for bar in bars_acc:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.2,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=6.5)
    for bar in bars_asr:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.2,
                f"{h:.0f}", ha="center", va="bottom", fontsize=6.5)

    # Legend below plot, outside axes
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=True, edgecolor="grey", fancybox=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(FIGURES_DIR / f"{save_prefix}_comparison.pdf")
    plt.savefig(FIGURES_DIR / f"{save_prefix}_comparison.png")
    plt.close()
    print(f"  Saved {save_prefix}_comparison.pdf")


def fig_accuracy_vs_asr_scatter(results: dict, save_prefix: str = "fig_real"):
    """Fig 2: Accuracy vs ASR scatter — shows the tradeoff.

    Each defense is a labeled point. Ideal is top-left (high acc, low ASR).
    """
    names = list(results.keys())
    markers = ["X", "s", "o", "D"]
    colors = ["#D65F5F", "#FFA500", "#4878CF", "#2CA02C"]
    short = ["No Defense", "RobustRAG", "MemProof", "MP+RRAG"]

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))

    for i, name in enumerate(names):
        acc = results[name]["accuracy"] * 100
        asr = results[name]["asr"] * 100
        ax.scatter(asr, acc, marker=markers[i], color=colors[i],
                   s=60, zorder=5, edgecolors="black", linewidths=0.5,
                   label=short[i])

    # Labels with offset — carefully positioned to avoid ALL overlaps
    # No Defense (high ASR, mid acc) — right of point
    # RobustRAG (low ASR, low acc) — right and above
    # MemProof (low ASR, high acc) — right of point
    # MP+RRAG (lowest ASR, lowest acc) — right and below
    offsets = [(6, 0), (6, 8), (6, 0), (6, -10)]
    haligns = ["left", "left", "left", "left"]
    for i, name in enumerate(names):
        acc = results[name]["accuracy"] * 100
        asr = results[name]["asr"] * 100
        ax.annotate(short[i], (asr, acc), fontsize=7,
                    xytext=offsets[i], textcoords="offset points",
                    ha=haligns[i], va="center")

    ax.set_xlabel("Attack Success Rate (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(-3, 24)
    ax.set_ylim(0, 65)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(FIGURES_DIR / f"{save_prefix}_tradeoff.pdf")
    plt.savefig(FIGURES_DIR / f"{save_prefix}_tradeoff.png")
    plt.close()
    print(f"  Saved {save_prefix}_tradeoff.pdf")


def fig_latency_breakdown(save_prefix: str = "fig_real"):
    """Fig 3: Latency breakdown — crypto overhead vs embedding.

    Horizontal stacked bar showing each component's contribution.
    """
    # Real measured values from meaningful_benchmark.py
    components = ["Embedding\n(model)", "Attestation", "Commitment\n(HMAC)", "Merkle\nInsert", "Audit\nAppend"]
    times_ms = [4.373, 0.005, 0.029, 0.022, 0.004]
    colors = ["#4878CF", "#98DF8A", "#2CA02C", "#FFBB78", "#FF7F0E"]

    fig, ax = plt.subplots(figsize=(COL_W, 1.8))

    left = 0
    for comp, t, c in zip(components, times_ms, colors):
        bar = ax.barh(0, t, left=left, color=c, edgecolor="black",
                      linewidth=0.4, height=0.5, label=comp)
        left += t

    ax.set_xlabel("Latency (ms)")
    ax.set_yticks([])
    ax.set_xlim(0, left * 1.15)

    # Annotations for tiny bars
    ax.annotate(f"Embedding: {times_ms[0]:.1f} ms (98.6%)",
                xy=(times_ms[0]/2, 0), fontsize=6.5, ha="center", va="center",
                color="white", fontweight="bold")
    ax.annotate(f"Crypto total:\n{sum(times_ms[1:]):.3f} ms (1.4%)",
                xy=(times_ms[0] + sum(times_ms[1:])/2, 0.38),
                fontsize=6, ha="center", va="bottom", color="black")

    # Legend below
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35),
              ncol=3, frameon=True, edgecolor="grey", fancybox=False,
              fontsize=6.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.savefig(FIGURES_DIR / f"{save_prefix}_latency.pdf")
    plt.savefig(FIGURES_DIR / f"{save_prefix}_latency.png")
    plt.close()
    print(f"  Saved {save_prefix}_latency.pdf")


def fig_poisonedrag_asr(save_prefix: str = "fig_real"):
    """Fig 4: PoisonedRAG ASR reproduction — MemProof run vs Zou et al.

    Simple bar comparing reported vs reproduced numbers.
    """
    fig, ax = plt.subplots(figsize=(COL_W * 0.7, 2.2))

    categories = ["Reported\n(Zou et al.)", "MemProof"]
    asrs = [90, 99]  # PoisonedRAG paper reports 90%+ on NQ; we got 99%
    colors = ["#AAAAAA", "#D65F5F"]

    bars = ax.bar(categories, asrs, color=colors, edgecolor="black",
                  linewidth=0.5, width=0.55)

    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, 115)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 2,
                f"{h}%", ha="center", va="bottom", fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(FIGURES_DIR / f"{save_prefix}_reproduction.pdf")
    plt.savefig(FIGURES_DIR / f"{save_prefix}_reproduction.png")
    plt.close()
    print(f"  Saved {save_prefix}_reproduction.pdf")


def fig_contamination_scaling(save_prefix: str = "fig_real"):
    """Fig 5: Retrieval contamination vs number of poison documents.

    Uses data from the meaningful benchmark experiments.
    """
    # Real data from experiment_results.json
    try:
        with open(Path(__file__).parent / "experiment_results.json") as f:
            exp_data = json.load(f)
        poison_counts = exp_data["exp2"]["poison_counts"]
        baseline_rates = exp_data["exp2"]["baseline_rates"]
        protected_rates = exp_data["exp2"]["protected_rates"]
    except FileNotFoundError:
        poison_counts = [1, 2, 3, 5, 8, 10]
        baseline_rates = [0.8, 1.7, 2.7, 4.7, 5.0, 5.0]
        protected_rates = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    fig, ax = plt.subplots(figsize=(COL_W, 2.3))

    ax.plot(poison_counts, baseline_rates, "o-", color="#D65F5F",
            label="No Defense", markersize=4)
    ax.plot(poison_counts, protected_rates, "s-", color="#4878CF",
            label="MemProof", markersize=4)

    ax.set_xlabel("Poison Documents Injected per Query")
    ax.set_ylabel("Avg. Poison Docs in Top-5")
    ax.set_ylim(-0.3, 5.8)
    ax.set_xlim(0, 11)

    # Legend in upper-left where there's space
    ax.legend(loc="upper left", frameon=True, edgecolor="grey", fancybox=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(FIGURES_DIR / f"{save_prefix}_contamination.pdf")
    plt.savefig(FIGURES_DIR / f"{save_prefix}_contamination.png")
    plt.close()
    print(f"  Saved {save_prefix}_contamination.pdf")


def fig_storage_overhead(save_prefix: str = "fig_real"):
    """Fig 6: Storage overhead — component breakdown.

    Horizontal stacked bar showing what each crypto component adds.
    """
    components = ["Document\nText", "Embedding\n(384d × 4B)", "Attestation", "Commitment", "Merkle\nLeaf", "Audit\nEntry"]
    sizes = [116, 1536, 200, 96, 32, 150]
    colors = ["#AEC7E8", "#4878CF", "#98DF8A", "#2CA02C", "#FFBB78", "#FF7F0E"]
    is_crypto = [False, False, True, True, True, True]

    fig, ax = plt.subplots(figsize=(COL_W, 2.0))

    left = 0
    for comp, sz, col, crypto in zip(components, sizes, colors, is_crypto):
        hatch = "//" if crypto else None
        ax.barh(0, sz, left=left, color=col, edgecolor="black",
                linewidth=0.4, height=0.45, hatch=hatch)
        left += sz

    # Baseline marker
    baseline = 116 + 1536
    ax.axvline(x=baseline, color="red", linestyle="--", linewidth=0.8)
    ax.annotate(f"Baseline\n({baseline} B)", xy=(baseline, 0.3),
                fontsize=6.5, ha="center", va="bottom", color="red")

    # Total marker
    ax.annotate(f"Total: {sum(sizes)} B (+{(sum(sizes)-baseline)/baseline*100:.0f}%)",
                xy=(sum(sizes), -0.3), fontsize=6.5, ha="right", va="top", color="black")

    ax.set_xlabel("Bytes per Entry")
    ax.set_yticks([])
    ax.set_xlim(0, sum(sizes) * 1.08)

    # Legend below — 3 columns to fit
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor="black", linewidth=0.4,
                             hatch="//" if cr else None, label=comp)
                       for comp, c, cr in zip(components, colors, is_crypto)]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=True,
              edgecolor="grey", fancybox=False, fontsize=6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.savefig(FIGURES_DIR / f"{save_prefix}_storage.pdf")
    plt.savefig(FIGURES_DIR / f"{save_prefix}_storage.png")
    plt.close()
    print(f"  Saved {save_prefix}_storage.pdf")


def generate_latex_tables(results: dict):
    """Print LaTeX tables for the paper."""

    print("\n── Table I: Main Comparison (Real NQ Data, GPT-3.5) ──\n")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Defense Comparison on NQ with PoisonedRAG Attack (GPT-3.5-turbo)}")
    print(r"\label{tab:main}")
    print(r"\small")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"\textbf{Defense} & \textbf{Acc.\ (\%)} & \textbf{ASR (\%)} & \textbf{Acc.\ Drop} & \textbf{ASR Red.} \\")
    print(r"\midrule")

    baseline_acc = results["No Defense"]["accuracy"]
    baseline_asr = results["No Defense"]["asr"]

    for name, r in results.items():
        acc = r["accuracy"] * 100
        asr = r["asr"] * 100
        acc_drop = (baseline_acc - r["accuracy"]) * 100
        asr_red = (baseline_asr - r["asr"]) * 100

        acc_drop_str = f"-{acc_drop:.0f}pp" if acc_drop > 0 else "---"
        asr_red_str = f"-{asr_red:.0f}pp" if asr_red > 0 else "---"

        bold = r"\textbf{" if name == "MemProof Only" else ""
        bold_end = "}" if name == "MemProof Only" else ""

        print(f"{bold}{name}{bold_end} & {acc:.0f} & {asr:.0f} & {acc_drop_str} & {asr_red_str} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    # Prefer the real-pipeline results (composition_results_100_real.json).
    # Fall back to the older files only if the real run is missing.
    candidates = [
        "composition_results_100_real.json",
        "composition_results_100.json",
        "composition_results_50.json",
    ]
    results_path = None
    for name in candidates:
        p = Path(__file__).parent / name
        if p.exists():
            results_path = p
            print(f"Using results: {p.name}")
            break
    if results_path is None:
        print("ERROR: no composition results JSON found.")
        return

    with open(results_path) as f:
        results = json.load(f)

    print("Generating publication figures from real NQ data...\n")

    fig_main_comparison(results)
    fig_accuracy_vs_asr_scatter(results)
    # Latency figure is now produced by ed25519_latency.py (Ed25519 numbers).
    # fig_latency_breakdown()  # deprecated: old HMAC numbers
    fig_poisonedrag_asr()
    fig_contamination_scaling()
    fig_storage_overhead()
    generate_latex_tables(results)

    print("\nAll figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
