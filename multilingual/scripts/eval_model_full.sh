#!/bin/bash
# Full (final-checkpoint) multilingual evaluation for one model. Runs every
# required eval in sequence:
#   1. zero-shot   (scripts/zeroshot_model.sh    — BLiMP, HellaSwag, Hanzi, ...)
#   2. Global PIQA (scripts/global_piqa_model.sh — acc_norm-scored, kept separate)
#   3. MECO        (scripts/meco_model.sh        — per-word surprisal, server-scored)
#   4. finetune    (scripts/finetune_model.sh    — per-task classification heads)
#
# Steps run independently; a failure in one does not abort the others
# (incomplete submissions are allowed — missing tasks are scored as 0).
#
# Run from multilingual/ (the script cd's there itself, so `bash
# scripts/eval_model_full.sh ...` works from anywhere).
#
# For the intermediate-checkpoint learning curves, run scripts/eval_model_fast.sh
# separately. When both are done, collate with:
#   python scripts/collate_results.py --model_name YOUR_MODEL --fast
model_name=""
langs="eng nld zho"
revision="main"
bos_fix=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name) model_name="$2"; shift 2 ;;
        --langs)      langs="$2";      shift 2 ;;
        --revision)   revision="$2";   shift 2 ;;
        --bos_fix)    bos_fix="$2";    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$model_name" ]]; then
    echo "Error: --model_name is required"; exit 1
fi

# Resolve to multilingual/ so the sub-scripts' relative paths (results/, tasks/,
# finetune/, scripts/run_lm_eval.py) resolve correctly.
cd "$(dirname "$0")/.." || exit 1

echo "=== [1/4] Zero-shot ==="
bash scripts/zeroshot_model.sh --model_name "$model_name" --langs "$langs" --revision "$revision" --bos_fix "$bos_fix"

echo "=== [2/4] Global PIQA ==="
bash scripts/global_piqa_model.sh --model_name "$model_name" --langs "$langs" --revision "$revision" --bos_fix "$bos_fix"

echo "=== [3/4] MECO ==="
bash scripts/meco_model.sh --model_name "$model_name" --langs "$langs" --revision "$revision"

echo "=== [4/4] Finetune ==="
# finetune_model.sh always trains from the model's main revision (no --revision flag).
bash scripts/finetune_model.sh --model_name "$model_name" --langs "$langs"

echo "=== Full evaluation complete. Collate with: ==="
echo "  python scripts/collate_results.py --model_name $model_name --fast"
