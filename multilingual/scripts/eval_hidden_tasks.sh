#!/bin/bash
# Evaluate ONLY the tasks newly added for the 2026 multilingual track, so you can
# re-run them without redoing the whole eval. Each runs for BOTH the final
# checkpoint (main revision) AND all intermediate checkpoints (chck_1M ..
# chck_1000M, the "fast_all" variant, for the learning curves):
#   1. Hanzi        (scripts/hanzi_model.sh       / hanzi_model_fast_all.sh
#                    — hanzi_structure, hanzi_pinyin; Chinese-only, self-skips
#                    unless zho is in --langs)
#   2. Global PIQA  (scripts/global_piqa_model.sh / global_piqa_model_fast_all.sh)
#   3. MECO         (scripts/meco_model.sh        / meco_model_fast_all.sh)
#   4. finetune POS (per-token classification on Universal Dependencies, final-only)
#
# This is a subset of scripts/eval_model_full.sh — it skips the standard
# zero-shot tasks (BLiMP, HellaSwag, ...) and the non-POS finetune tasks.
#
# Steps run independently; a failure in one does not abort the others.
# Run from multilingual/ (the script cd's there itself, so `bash
# scripts/eval_hidden_tasks.sh ...` works from anywhere).
#
# When done, collate with:
#   python scripts/collate_results.py --model_name YOUR_MODEL --fast

# --- shared flags ---
model_name=""
langs="eng nld zho"
bos_fix=1
# --- finetune (POS) hyperparameters — defaults mirror scripts/finetune_model.sh ---
LR=5e-5
PATIENCE=3
BSZ=64
MAX_EPOCHS=10
SEED=12

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name) model_name="$2"; shift 2 ;;
        --langs)      langs="$2";      shift 2 ;;
        --bos_fix)    bos_fix="$2";    shift 2 ;;
        --lr)         LR="$2";         shift 2 ;;
        --patience)   PATIENCE="$2";   shift 2 ;;
        --bsz)        BSZ="$2";        shift 2 ;;
        --max_epochs) MAX_EPOCHS="$2"; shift 2 ;;
        --seed)       SEED="$2";       shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$model_name" ]]; then
    echo "Error: --model_name is required"; exit 1
fi

# Resolve to multilingual/ so relative paths (results/, tasks/, finetune/,
# scripts/run_lm_eval.py) resolve correctly.
cd "$(dirname "$0")/.." || exit 1

# --- 1. Hanzi (Chinese only; the sub-scripts self-skip when zho not requested) --
echo "=== [1/7] Hanzi (final checkpoint) ==="
bash scripts/hanzi_model.sh --model_name "$model_name" --langs "$langs" --revision "main" --bos_fix "$bos_fix"

echo "=== [2/7] Hanzi (all checkpoints) ==="
bash scripts/hanzi_model_fast_all.sh --model_name "$model_name" --langs "$langs" --bos_fix "$bos_fix"

# --- 2. Global PIQA ------------------------------------------------------------
echo "=== [3/7] Global PIQA (final checkpoint) ==="
bash scripts/global_piqa_model.sh --model_name "$model_name" --langs "$langs" --revision "main" --bos_fix "$bos_fix"

echo "=== [4/7] Global PIQA (all checkpoints) ==="
bash scripts/global_piqa_model_fast_all.sh --model_name "$model_name" --langs "$langs" --bos_fix "$bos_fix"

# --- 3. MECO -------------------------------------------------------------------
echo "=== [5/7] MECO (final checkpoint) ==="
bash scripts/meco_model.sh --model_name "$model_name" --langs "$langs" --revision "main"

echo "=== [6/7] MECO (all checkpoints) ==="
bash scripts/meco_model_fast_all.sh --model_name "$model_name" --langs "$langs"

# --- 4. finetune POS -----------------------------------------------------------
# Mirrors the POS step of scripts/finetune_model.sh: one model trained on a
# uniform cross-lingual mixture of all requested languages, evaluated per
# language (results in finetune/results/<model>/pos/<lang>/).
echo "=== [7/7] Finetune POS (joint cross-lingual) ==="
model_basename=$(basename "$model_name")
pos_langs=""
for LANGUAGE in $langs; do pos_langs="$pos_langs ${LANGUAGE:0:2}"; done
pos_langs="${pos_langs# }"  # eng/nld/zho -> "en nl zh"
mkdir -p "finetune/results/${model_basename}/pos/"
python3 finetune/finetune_token_classification.py \
      --model_name_or_path "$model_name" \
      --language "$pos_langs" \
      --output_dir "finetune/results/${model_basename}/pos" \
      --do_train \
      --do_eval \
      --do_predict \
      --max_seq_length 128 \
      --per_device_train_batch_size "$BSZ" \
      --learning_rate "$LR" \
      --num_train_epochs "$MAX_EPOCHS" \
      --patience "$PATIENCE" \
      --eval_strategy epoch \
      --save_strategy epoch \
      --overwrite_output_dir \
      --seed "$SEED"

echo "=== Hidden-task evaluation complete. Collate with: ==="
echo "  python scripts/collate_results.py --model_name $model_name --fast"
