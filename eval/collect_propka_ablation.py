"""Collect PROPKA ablation results and generate summary table + bar chart.

Reads from results_propka_ablation_v2/<config>/bilstm_4fold_results.json
All configs (including full) are loaded from actual training results.

Outputs:
  - results_propka_ablation_v2/ablation_summary.json
  - figures/fig_propka_ablation.pdf
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results_propka_ablation_v2")
OUT_FIG = Path("figures")
OUT_FIG.mkdir(parents=True, exist_ok=True)

# Config order and display names
CONFIGS = [
    ("full",           "Full (IF1+PROPKA)",  518),
    ("no_pka",         r"$-$p$K_\mathrm{a}$", 517),
    ("no_desolvation", r"$-$Desolvation",     517),
    ("no_bb_hbond",    r"$-$BB H-bond",       517),
    ("no_sc_hbond",    r"$-$SC H-bond",       517),
    ("no_coulomb",     r"$-$Coulombic",       517),
    ("no_combined",    r"$-$Combined",        517),
    ("if1_only",       "IF1 only",            512),
]


def load_results():
    rows = []

    for config_name, display, expected_dim in CONFIGS:
        path = RESULTS_DIR / config_name / "bilstm_4fold_results.json"
        if not path.exists():
            print(f"  [MISSING] {config_name}")
            continue

        with open(path) as f:
            d = json.load(f)

        # Per-fold test MCC
        fold_mccs = [r["test_metrics"]["mcc"] for r in d["fold_results"]
                     if "test_metrics" in r]
        if not fold_mccs:
            fold_mccs = [r["val_metrics"]["mcc"] for r in d["fold_results"]]

        row = {
            "config": config_name,
            "display": display,
            "input_dim": d.get("args", {}).get("input_dim", expected_dim),
            "avg_test_mcc": np.mean(fold_mccs),
            "std_test_mcc": np.std(fold_mccs),
            "ensemble_test_mcc": d.get("ensemble_test_mcc", np.mean(fold_mccs)),
            "fold_mccs": fold_mccs,
        }

        # Additional metrics
        for k in ["accuracy", "precision", "recall", "f1"]:
            vals = [r.get("test_metrics", r["val_metrics"])[k]
                    for r in d["fold_results"]]
            row[f"avg_{k}"] = np.mean(vals)
            row[f"std_{k}"] = np.std(vals)

        # Ensemble metrics
        ens = d.get("ensemble_test_metrics", {})
        if ens:
            for k in ["accuracy", "precision", "recall", "f1"]:
                row[f"ens_{k}"] = ens.get(k, row[f"avg_{k}"])

        rows.append(row)

    return rows


def print_table(rows):
    full_mcc = None
    full_ens = None
    for r in rows:
        if r["config"] == "full":
            full_mcc = r["avg_test_mcc"]
            full_ens = r["ensemble_test_mcc"]
            break

    print("\n" + "=" * 90)
    print("PROPKA Ablation Study v2 — Summary")
    print("=" * 90)
    print(f"{'Config':<20} {'Dim':>4} {'FoldAvg':>8} {'Ensemble':>9} "
          f"{'dMCC(ens)':>10} {'F1':>7}")
    print("-" * 90)
    for r in rows:
        delta = r["ensemble_test_mcc"] - full_ens if full_ens else 0
        sign = "+" if delta >= 0 else ""
        f1 = r.get("ens_f1", r["avg_f1"])
        print(f"{r['config']:<20} {r['input_dim']:>4} "
              f"{r['avg_test_mcc']:>8.4f} {r['ensemble_test_mcc']:>9.4f} "
              f"{sign}{delta:>9.4f} {f1:>7.4f}")
    print("=" * 90)


def make_figure(rows):
    """Bar chart: delta ensemble MCC relative to full model."""
    full_ens = None
    for r in rows:
        if r["config"] == "full":
            full_ens = r["ensemble_test_mcc"]
            break
    if full_ens is None:
        print("Cannot generate figure: full config results missing")
        return

    ablated = [r for r in rows if r["config"] != "full"]

    names = [r["display"] for r in ablated]
    deltas = [r["ensemble_test_mcc"] - full_ens for r in ablated]
    stds = [r["std_test_mcc"] for r in ablated]
    colors = ["#C62828" if d < -0.005 else "#1565C0" if d > 0.005 else "#757575"
              for d in deltas]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(range(len(names)), deltas, xerr=stds,
                   color=colors, edgecolor="white", linewidth=0.5,
                   capsize=3, height=0.6)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel(r"$\Delta$MCC (relative to full model)", fontsize=11)
    ax.set_title(f"PROPKA Feature Ablation (Full model MCC = {full_ens:.3f})",
                 fontsize=12, fontweight="bold")

    for i, (d, s) in enumerate(zip(deltas, stds)):
        sign = "+" if d >= 0 else ""
        ax.text(d + (s + 0.003) * (1 if d >= 0 else -1), i,
                f"{sign}{d:.3f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out_path = OUT_FIG / "fig_propka_ablation.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")
    plt.close()


def save_summary(rows):
    summary = {
        "configs": [{k: v for k, v in r.items() if k != "fold_mccs"}
                    for r in rows],
        "fold_details": {r["config"]: r["fold_mccs"] for r in rows},
    }
    out_path = RESULTS_DIR / "ablation_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved: {out_path}")


def generate_latex_table(rows):
    """Print a LaTeX table snippet."""
    full_ens = None
    for r in rows:
        if r["config"] == "full":
            full_ens = r["ensemble_test_mcc"]
            break

    print("\n% ── LaTeX table snippet ──")
    print(r"\begin{table}[!t]")
    print(r"\processtable{PROPKA feature ablation ($w=11$, 4-fold CV).\label{tab:propka_ablation}}")
    print(r"{\begin{tabular}{@{}lccccc@{}}")
    print(r"\toprule")
    print(r"Configuration & Dim & MCC & $\Delta$MCC & F1 & Recall \\")
    print(r"\midrule")

    for r in rows:
        delta = r["ensemble_test_mcc"] - full_ens if full_ens else 0
        mcc_str = f"{r['ensemble_test_mcc']:.3f}"
        f1 = r.get("ens_f1", r["avg_f1"])
        rec = r.get("ens_recall", r["avg_recall"])
        f1_str = f"{f1:.3f}"
        rec_str = f"{rec:.3f}"
        dim = r["input_dim"]

        if r["config"] == "full":
            print(f"\\textbf{{Full (IF1+PROPKA)}} & \\textbf{{{dim}}} & "
                  f"\\textbf{{{mcc_str}}} & --- & "
                  f"\\textbf{{{f1_str}}} & \\textbf{{{rec_str}}} \\\\")
        else:
            d_str = f"{delta:+.3f}"
            name = r["display"]
            if r["config"] == "if1_only":
                name = "IF1 only"
            print(f"{name} & {dim} & {mcc_str} & {d_str} & {f1_str} & {rec_str} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"{MCC: ensemble of 4-fold models. "
          r"$\Delta$MCC: change relative to the full model. "
          r"Negative values indicate performance degradation upon feature removal.}")
    print(r"\end{table}")


if __name__ == "__main__":
    rows = load_results()
    if not rows:
        print("No results found. Run train/run_propka_ablation.sh first.")
        exit(1)

    print_table(rows)
    make_figure(rows)
    save_summary(rows)
    generate_latex_table(rows)
