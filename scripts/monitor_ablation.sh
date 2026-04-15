#!/bin/bash
# Quick status check for ablation runs (v2)
# Usage: bash train/monitor_ablation.sh

BASE="results_propka_ablation_v2"
CONFIGS=(full no_pka no_desolvation no_bb_hbond no_sc_hbond no_coulomb no_combined if1_only)

echo "═══════════════════════════════════════════════════════════════════"
echo "  PROPKA Ablation v2 — Status @ $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Data: data_if1_w11 | bs=256, lr=0.002, dropout=0.4, wd=0.005"
echo "═══════════════════════════════════════════════════════════════════"
printf "%-17s | %-8s | %-10s | %-10s | %s\n" "Config" "Status" "Fold MCC" "Ens MCC" "Progress"
echo "──────────────────────────────────────────────────────────────────"

for cfg in "${CONFIGS[@]}"; do
    dir="${BASE}/${cfg}"
    if [ -f "${dir}/bilstm_4fold_results.json" ]; then
        read -r avg_mcc ens_mcc <<< $(python3 -c "
import json; d=json.load(open('${dir}/bilstm_4fold_results.json'))
avg = d.get('avg_test_mcc', 0)
ens = d.get('ensemble_test_mcc', 0)
print(f'{avg:.4f} {ens:.4f}')
" 2>/dev/null)
        printf "%-17s | %-8s | %-10s | %-10s | %s\n" "$cfg" "DONE" "$avg_mcc" "$ens_mcc" "—"
    elif [ -f "${dir}/train.log" ]; then
        last_fold=$(grep -oP 'Fold \d+/\d+' "${dir}/train.log" 2>/dev/null | tail -1)
        last_epoch=$(grep -oP 'Epoch \d+/\d+' "${dir}/train.log" 2>/dev/null | tail -1)
        last_mcc=$(grep -oP 'Val MCC: [\d.]+' "${dir}/train.log" 2>/dev/null | tail -1)
        printf "%-17s | %-8s | %-10s | %-10s | %s %s %s\n" "$cfg" "RUNNING" "—" "—" "$last_fold" "$last_epoch" "$last_mcc"
    else
        printf "%-17s | %-8s | %-10s | %-10s | %s\n" "$cfg" "PENDING" "—" "—" ""
    fi
done
echo "═══════════════════════════════════════════════════════════════════"
