#!/bin/bash
# Fast multilingual evaluation: zero-shot tasks across all intermediate
# checkpoints (chck_1M .. chck_1000M), for the learning-curve submission.
#   1. zero-shot   (scripts/zeroshot_model_fast_all.sh)
#   2. Global PIQA (scripts/global_piqa_model_fast_all.sh)
#   3. MECO        (scripts/meco_model.sh, once per checkpoint)
#
# Hanzi runs inside the zero-shot Chinese group at every checkpoint. MECO is
# also evaluated at every checkpoint below. Finetune remains final-only.
#
# Run from multilingual/ (the script cd's there itself, so `bash
# scripts/eval_model_fast.sh ...` works from anywhere).
#
# When both fast and full evals are done, collate with:
#   python scripts/collate_results.py --model_name YOUR_MODEL --fast
model_name=""
langs="eng nld zho"
bos_fix=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name) model_name="$2"; shift 2 ;;
        --langs)      langs="$2";      shift 2 ;;
        --bos_fix)    bos_fix="$2";    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$model_name" ]]; then
    echo "Error: --model_name is required"; exit 1
fi

# Resolve to multilingual/ so the sub-scripts' relative paths resolve correctly.
cd "$(dirname "$0")/.." || exit 1

echo "=== [1/3] Zero-shot, including Hanzi (all checkpoints) ==="
bash scripts/zeroshot_model_fast_all.sh --model_name "$model_name" --langs "$langs" --bos_fix "$bos_fix"

echo "=== [2/3] Global PIQA (all checkpoints) ==="
bash scripts/global_piqa_model_fast_all.sh --model_name "$model_name" --langs "$langs" --bos_fix "$bos_fix"

echo "=== [3/3] MECO (all checkpoints) ==="
bash scripts/meco_model_fast_all.sh --model_name "$model_name" --langs "$langs"

echo "=== Fast evaluation complete. Collate with: ==="
echo "  python scripts/collate_results.py --model_name $model_name --fast"
