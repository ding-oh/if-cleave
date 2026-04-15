"""Regenerate ALL figures using w=11 model results.

Reads from cd4_validation_w11/ and outputs to figures/.
"""

import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# ── Configuration ──
BASE = Path("cd4_validation_w11")
OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

TARGETS_MAIN = ["influenza_ha", "vatreptacog", "rsv_f"]
TARGETS_ALL = ["influenza_ha", "vatreptacog", "rsv_f", "spike"]

META = {
    "influenza_ha": {"label": "Influenza HA", "short": "HA", "color": "#7B1FA2", "pdb": "1RU7"},
    "vatreptacog": {"label": "Factor VIIa", "short": "FVIIa", "color": "#E65100", "pdb": "1DAN"},
    "rsv_f": {"label": "RSV F", "short": "RSV F", "color": "#2E7D32", "pdb": "8W3K"},
    "spike": {"label": "Spike", "short": "Spike", "color": "#1565C0", "pdb": "6ZGE"},
}


def load_data(tag):
    pred_path = BASE / tag / "predictions" / f"{tag}_predictions.csv"
    report_path = BASE / tag / "analysis" / "enrichment_report.json"
    probs, is_bd, min_dists, positions, residues = [], [], [], [], []
    with open(pred_path) as f:
        for row in csv.DictReader(f):
            positions.append(int(row["position"]))
            residues.append(row["residue"])
            probs.append(float(row["probability"]))
            is_bd.append(int(row["is_boundary"]))
            min_dists.append(int(row["min_dist_to_boundary"]))
    with open(report_path) as f:
        report = json.load(f)
    return {
        "positions": np.array(positions), "residues": residues,
        "probs": np.array(probs), "is_boundary": np.array(is_bd),
        "min_dists": np.array(min_dists), "report": report,
    }


def sig_str(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "n.s."


# ══════════════════════════════════════════════════════════════
#  Fig 1: Cleavage profiles (4 panels)
# ══════════════════════════════════════════════════════════════
def fig1_profiles():
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=False)
    for ax, tag in zip(axes, TARGETS_MAIN):
        m = META[tag]
        d = load_data(tag)
        color = m["color"]

        ax.fill_between(d["positions"], d["probs"], alpha=0.15, color=color)
        ax.plot(d["positions"], d["probs"], color=color, linewidth=0.6)

        bd_idx = np.where(d["is_boundary"] == 1)[0]
        for bi in bd_idx:
            ax.axvline(d["positions"][bi], color="#E53935", alpha=0.2, linewidth=0.4, linestyle="--")
        ax.axhline(0.5, color="gray", linewidth=0.5, linestyle=":", alpha=0.4)

        r = d["report"]
        p_val = r["enrichment"]["p_value"]
        title = f"{m['short']} (PDB {m['pdb']})  |  p={p_val:.4f} {sig_str(p_val)}"
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.set_ylabel("P(cleavage)", fontsize=9)
        ax.set_ylim(-0.02, 1.05)

    axes[-1].set_xlabel("PDB Residue Position", fontsize=10)
    plt.suptitle("IF-Cleave Cleavage Profiles with Elution Epitope Boundaries ($w=11$)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT / "fig1_profiles.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT / "fig1_profiles.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Fig 1 saved")


# ══════════════════════════════════════════════════════════════
#  Fig 2: Distance decay (2x2)
# ══════════════════════════════════════════════════════════════
def fig2_distance():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, tag in zip(axes.flat, TARGETS_MAIN):
        m = META[tag]
        d = load_data(tag)
        dp = d["report"]["distance_profile"]
        bins = np.array(dp["bins"])
        means = np.array([float(x) if not isinstance(x, str) and not np.isnan(float(x)) else np.nan
                          for x in dp["mean_prob"]])
        rho = dp["spearman_rho"]
        color = m["color"]

        bar_colors = [color if d <= 2 else matplotlib.colors.to_rgba(color, 0.5) for d in bins]
        ax.bar(bins, means, color=bar_colors, edgecolor="white", linewidth=0.3, width=0.8)

        valid = ~np.isnan(means)
        if valid.sum() > 2:
            z = np.polyfit(bins[valid], means[valid], 1)
            ax.plot(bins, np.poly1d(z)(bins), color="#E53935", linewidth=1.2, alpha=0.6)

        rho_str = f"ρ = {rho:.3f}" if not (isinstance(rho, float) and np.isnan(rho)) and rho != "NaN" else "ρ = —"
        ax.set_title(f"{m['short']}  ({rho_str})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Distance to boundary", fontsize=9)
        ax.set_ylabel("Mean P(cleavage)", fontsize=9)
        ax.set_ylim(0, min(1.0, np.nanmax(means) * 1.3) if valid.any() else 1.0)

    plt.suptitle("Distance–Probability Decay ($w=11$)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT / "fig2_distance.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT / "fig2_distance.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Fig 2 saved")


# ══════════════════════════════════════════════════════════════
#  Fig 3: ROC curves
# ══════════════════════════════════════════════════════════════
def fig3_roc():
    fig, ax = plt.subplots(figsize=(7, 7))
    for tag in TARGETS_MAIN:
        m = META[tag]
        d = load_data(tag)
        fpr, tpr, _ = roc_curve(d["is_boundary"], d["probs"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=m["color"], linewidth=2.5,
                label=f"{m['short']} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC: Epitope Boundary Detection ($w=11$)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUT / "fig3_roc.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT / "fig3_roc.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Fig 3 saved")


# ══════════════════════════════════════════════════════════════
#  Fig 4: Enrichment bars
# ══════════════════════════════════════════════════════════════
def fig4_enrichment():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = ["effect_size", "spearman_rho", "auc"]
    titles = ["Effect Size", "|Spearman ρ|", "Boundary AUC"]

    for ax, metric, title in zip(axes, metrics, titles):
        vals, colors, labels = [], [], []
        for tag in TARGETS_MAIN:
            m = META[tag]
            d = load_data(tag)
            r = d["report"]
            if metric == "effect_size":
                v = r["enrichment"]["effect_size"]
            elif metric == "spearman_rho":
                v = abs(r["distance_profile"]["spearman_rho"]) if not isinstance(
                    r["distance_profile"]["spearman_rho"], str) else 0
            else:
                v = r["auc"]
            vals.append(v if not np.isnan(v) else 0)
            colors.append(m["color"])
            labels.append(m["short"])

        bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 1)

    plt.suptitle("Enrichment Metrics ($w=11$)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUT / "fig4_enrichment.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT / "fig4_enrichment.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Fig 4 saved")


# ══════════════════════════════════════════════════════════════
#  Fig 5: Control experiment bars
# ══════════════════════════════════════════════════════════════
def fig5_control():
    # Load all-IEDB enrichment results from cd4_validation_folds (w=7 baseline)
    # and elution from w=11
    # For the control, the key comparison is elution vs all-IEDB
    # We only have all-IEDB results from w=7 model, which is acceptable for the control
    # since the point is that synthetic boundaries don't show enrichment regardless of model

    control_data = {
        "influenza_ha": {"all_p": 0.206, "elution_p": 0.0001},
        "vatreptacog": {"all_p": 0.120, "elution_p": 0.0003},
        "rsv_f": {"all_p": 0.131, "elution_p": 0.005},
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(control_data))
    width = 0.35
    labels = [META[t]["short"] for t in control_data]

    all_vals = [-np.log10(v["all_p"]) for v in control_data.values()]
    elu_vals = [-np.log10(v["elution_p"]) for v in control_data.values()]

    ax.bar(x - width/2, all_vals, width, color="#BDBDBD", edgecolor="white", label="All IEDB (incl. synthetic)")
    ax.bar(x + width/2, elu_vals, width, color="#E53935", edgecolor="white", label="Elution only (natural)")

    ax.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=1, alpha=0.5, label="p = 0.05")
    ax.axhline(-np.log10(0.001), color="gray", linestyle=":", linewidth=1, alpha=0.5, label="p = 0.001")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("$-\\log_{10}(p)$", fontsize=11)
    ax.set_title("Control: Synthetic vs. Naturally Processed Boundaries", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT / "fig5_control.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT / "fig5_control.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Fig 5 saved")


# ══════════════════════════════════════════════════════════════
#  Fig 6: Threshold / Operating modes (HA case study)
# ══════════════════════════════════════════════════════════════
def fig6_threshold():
    d = load_data("influenza_ha")
    probs, is_bd, min_dists = d["probs"], d["is_boundary"], d["min_dists"]
    positions = d["positions"]
    n = len(probs)

    modes = {
        "Balanced":  {"t": 0.50, "color": "#78909C"},
        "Sensitive": {"t": 0.30, "color": "#E53935"},
        "Screening": {"t": 0.10, "color": "#1E88E5"},
    }

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 1, figure=fig, hspace=0.35, height_ratios=[1.2, 1])

    # Panel A: Profile with threshold lines
    ax = fig.add_subplot(gs[0])
    ax.fill_between(positions, probs, alpha=0.15, color="#7B1FA2")
    ax.plot(positions, probs, color="#7B1FA2", linewidth=0.8)
    bd_idx = np.where(is_bd == 1)[0]
    for bi in bd_idx:
        ax.axvline(positions[bi], color="#E53935", alpha=0.2, linewidth=0.4, linestyle="--")
    for name, cfg in modes.items():
        ax.axhline(cfg["t"], color=cfg["color"], linewidth=1.5, alpha=0.7, label=f"{name} (t={cfg['t']:.2f})")

    # Mark missed at balanced, recovered at sensitive
    missed = is_bd & (probs < 0.50)
    recovered = is_bd & (probs >= 0.30) & (probs < 0.50)
    m_idx = np.where(missed)[0]
    r_idx = np.where(recovered)[0]
    if len(m_idx) > 0:
        ax.scatter(positions[m_idx], probs[m_idx], color="#E53935", s=40, marker="x", linewidths=2, zorder=5,
                   label=f"Missed in Balanced (n={len(m_idx)})")
    if len(r_idx) > 0:
        ax.scatter(positions[r_idx], probs[r_idx], color="#43A047", s=40, marker="^", linewidths=1.5, zorder=5,
                   label=f"Recovered in Sensitive (n={len(r_idx)})")

    ax.set_title("(A) Influenza HA — Threshold-Adaptive Boundary Detection", fontsize=11, fontweight="bold")
    ax.set_xlabel("PDB Residue Position (1RU7)", fontsize=10)
    ax.set_ylabel("P(cleavage)", fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, loc="upper right", ncol=2)

    # Panel B: Recall/NPV vs threshold
    ax2 = fig.add_subplot(gs[1])
    thresholds = np.linspace(0.01, 0.99, 200)
    recalls, npvs = [], []
    for t in thresholds:
        pred = probs >= t
        tp = (pred & (is_bd == 1)).sum()
        fn = (~pred & (is_bd == 1)).sum()
        tn = (~pred & (is_bd == 0)).sum()
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        npvs.append(tn / (tn + fn) if (tn + fn) > 0 else 0)
    ax2.plot(thresholds, recalls, color="#E53935", linewidth=2.5, label="Recall")
    ax2.plot(thresholds, npvs, color="#1E88E5", linewidth=2.5, label="NPV")
    for name, cfg in modes.items():
        ax2.axvline(cfg["t"], color=cfg["color"], linewidth=1.5, linestyle="--", alpha=0.7)
    ax2.set_xlabel("Decision Threshold", fontsize=10)
    ax2.set_ylabel("Metric", fontsize=10)
    ax2.set_title("(B) Recall & NPV vs. Threshold", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.1)

    plt.suptitle("IF-Cleave Operating Modes — Threshold-Adaptive Prediction ($w=11$)",
                 fontsize=13, fontweight="bold")
    plt.savefig(OUT / "fig6_threshold.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT / "fig6_threshold.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Fig 6 saved")


# ══════════════════════════════════════════════════════════════
#  Fig S1: Window ablation
# ══════════════════════════════════════════════════════════════
def figS1_window_ablation():
    windows = [1, 3, 5, 7, 9, 11, 13, 15]
    mccs = [0.243, 0.245, 0.247, 0.256, 0.249, 0.260, 0.253, 0.253]
    f1s = [0.455, 0.473, 0.494, 0.514, 0.526, 0.553, 0.557, 0.571]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "#1565C0"
    color2 = "#E65100"

    ax1.bar([w - 0.3 for w in windows], mccs, width=0.5, color=color1, alpha=0.8, label="Ensemble MCC")
    ax1.set_xlabel("Window Size ($w$)", fontsize=11)
    ax1.set_ylabel("Ensemble Test MCC", fontsize=11, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(windows)
    ax1.set_ylim(0.23, 0.27)

    # Highlight best
    best_idx = mccs.index(max(mccs))
    ax1.bar(windows[best_idx] - 0.3, mccs[best_idx], width=0.5, color="#E53935", alpha=0.9)
    ax1.annotate(f"Best: $w$={windows[best_idx]}\nMCC={mccs[best_idx]:.3f}",
                 xy=(windows[best_idx], mccs[best_idx]),
                 xytext=(windows[best_idx] + 1.5, mccs[best_idx] + 0.005),
                 fontsize=9, fontweight="bold", color="#E53935",
                 arrowprops=dict(arrowstyle="->", color="#E53935"))

    ax2 = ax1.twinx()
    ax2.plot(windows, f1s, color=color2, linewidth=2, marker="o", markersize=6, label="Test F1")
    ax2.set_ylabel("Test F1", fontsize=11, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0.4, 0.6)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    ax1.set_title("Window Size Ablation Study", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "figS1_window_ablation.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT / "figS1_window_ablation.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Fig S1 saved")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Regenerating ALL figures (w=11 model)")
    print("=" * 60)
    fig1_profiles()
    fig2_distance()
    fig3_roc()
    fig4_enrichment()
    fig5_control()
    fig6_threshold()
    figS1_window_ablation()
    print(f"\nAll figures saved to {OUT}/")
    print(f"Files: {sorted(f.name for f in OUT.iterdir())}")
