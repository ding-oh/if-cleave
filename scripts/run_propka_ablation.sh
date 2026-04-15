#!/bin/bash
# ────────────────────────────────────────────────────────
#  PROPKA Leave-One-Out Ablation Study (v2)
#  8 configurations × 4-fold CV
#
#  Fixed from v1:
#    - Uses data_if1_w11 (window=11, matching paper)
#    - Matches paper hyperparameters (dropout=0.4, wd=0.005, etc.)
#    - Includes "full" baseline for fair comparison
#    - Computes ensemble MCC
#    - batch_size=128, lr=0.002 (scaled for speed)
#
#  Usage:  bash train/run_propka_ablation.sh [GPU_ID]
#  Default GPU: 0
# ────────────────────────────────────────────────────────
set -euo pipefail

GPU="${1:-0}"
DATA_DIR="data_if1_w11"
BASE_OUT="results_propka_ablation_v2"
COMMON="--model bilstm --hidden_dim 256 --dropout 0.4 \
        --epochs 500 --batch_size 256 --lr 0.002 --weight_decay 0.005 \
        --loss bce --patience 20 --n_folds 4 --seed 42 \
        --label_smoothing 0.05 --grad_clip 1.0 \
        --num_workers 4 --gpu ${GPU} --data_dir ${DATA_DIR}"

mkdir -p "${BASE_OUT}"

# ── Define ablation configs ────────────────────────────
declare -A CONFIGS
CONFIGS=(
    ["full"]=""
    ["no_pka"]="pka"
    ["no_desolvation"]="desolvation"
    ["no_bb_hbond"]="bb_hbond"
    ["no_sc_hbond"]="sc_hbond"
    ["no_coulomb"]="coulomb"
    ["no_combined"]="combined"
    ["if1_only"]="all_propka"
)

# Run list — full first as baseline, then ablations
ORDER=(full no_pka no_desolvation no_bb_hbond no_sc_hbond no_coulomb no_combined if1_only)

echo "========================================================"
echo "  PROPKA Ablation Study v2 — ${#ORDER[@]} configurations"
echo "  GPU: ${GPU}  |  Data: ${DATA_DIR}"
echo "  Hyperparams: bs=256, lr=0.002, dropout=0.4, wd=0.005"
echo "  label_smoothing=0.05, grad_clip=1.0, patience=20"
echo "========================================================"

for config_name in "${ORDER[@]}"; do
    exclude="${CONFIGS[$config_name]}"
    out_dir="${BASE_OUT}/${config_name}"
    mkdir -p "${out_dir}"

    echo ""
    echo "────────────────────────────────────────────────────"
    echo "  Config: ${config_name}"
    if [ -n "${exclude}" ]; then
        echo "  Excluding: ${exclude}"
    else
        echo "  Excluding: (none — full model)"
    fi
    echo "  Output: ${out_dir}"
    echo "────────────────────────────────────────────────────"

    # Skip if results already exist
    if ls "${out_dir}"/bilstm_4fold_results.json &>/dev/null; then
        echo "  [SKIP] Results already exist. Delete ${out_dir} to re-run."
        continue
    fi

    EXCLUDE_FLAG=""
    if [ -n "${exclude}" ]; then
        EXCLUDE_FLAG="--exclude_propka ${exclude}"
    fi

    START_T=$(date +%s)
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"

    python train/train_kfold.py ${COMMON} \
        --output_dir "${out_dir}" \
        ${EXCLUDE_FLAG} \
        2>&1 | tee "${out_dir}/train.log"

    END_T=$(date +%s)
    ELAPSED=$(( (END_T - START_T) / 60 ))
    echo "  [DONE] ${config_name} — ${ELAPSED} min ($(date '+%H:%M:%S'))"
done

echo ""
echo "========================================================"
echo "  All ablation runs complete!"
echo "  Results in: ${BASE_OUT}/"
echo "========================================================"

# ── Summary table ──────────────────────────────────────
echo ""
echo "Config            | input_dim | Avg Test MCC | Ensemble MCC"
echo "------------------|-----------|--------------|-------------"
for config_name in "${ORDER[@]}"; do
    out_dir="${BASE_OUT}/${config_name}"
    if [ -f "${out_dir}/bilstm_4fold_results.json" ]; then
        python3 -c "
import json
d = json.load(open('${out_dir}/bilstm_4fold_results.json'))
dim = d.get('args',{}).get('input_dim', '?')
avg_mcc = d.get('avg_test_mcc', 0)
ens_mcc = d.get('ensemble_test_mcc', 0)
print(f'${config_name:<17s} | {dim:>9} | {avg_mcc:>12.4f} | {ens_mcc:>12.4f}')
" 2>/dev/null || printf "%-17s | %9s | %s\n" "${config_name}" "?" "ERROR"
    else
        printf "%-17s | %9s | %12s | %s\n" "${config_name}" "?" "PENDING" ""
    fi
done
