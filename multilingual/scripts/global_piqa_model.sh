#!/bin/bash
# Launch the (hidden) Global PIQA tasks for a model, one language at a time.
# Kept separate from zeroshot_model.sh because Global PIQA is scored with acc_norm
# (length-normalized) only — never plain acc — so its numbers can never be confused
# with the acc-based zeroshot tasks or the leaderboard.
#
# These BabyLM tokenizers auto-append BOTH <s> and </s> on every encode(). The trailing </s>
# throws off lm-eval's context/continuation split (dropping the real first continuation token and
# scoring a spurious </s>); the leading <s> is wanted (the strict harness scores completions
# conditioned on <s>). The in-repo `hf-bos` model (scripts/bos_hf_model.py, run via
# scripts/run_lm_eval.py) fixes this at the code level: it tokenizes with add_special_tokens=False
# and prepends a single BOS, identical to the strict harness, with no tokenizer artifact on disk.
# This is opt-in via --bos_fix (default 1); participants whose tokenizer is already BOS-only (or
# assumes no BOS) can pass --bos_fix 0 to use the stock `--model hf`. Combined with the token-count
# acc_norm in tasks/global_piqa_*/utils.py (which reads GLOBAL_PIQA_TOKENIZER), this makes lm-eval
# reproduce the strict harness exactly.
model_name=""
langs="eng nld zho"
revision="main"
bos_fix=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name) model_name="$2"; shift 2 ;;
        --langs) langs="$2"; shift 2 ;;
        --revision) revision="$2"; shift 2 ;;
        --bos_fix) bos_fix="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$model_name" ]]; then
    echo "Error: --model_name is required"; exit 1
fi

# Select the BOS-fix model (in-repo hf-bos) or the stock hf model. The token-count metric encodes
# with add_special_tokens=False, so it counts correctly against the model's own tokenizer either way.
if [[ "$bos_fix" == "1" ]]; then
    runner="python3 scripts/run_lm_eval.py"; model_flag="hf-bos"; extra=",add_bos_token=True"
else
    runner="python3 -m lm_eval"; model_flag="hf"; extra=""
fi
export GLOBAL_PIQA_TOKENIZER="$model_name"

for lang in $langs; do
    task_name="global_piqa_${lang}"
    echo "Evaluating Global PIQA ${lang} (revision: ${revision})"
    $runner \
        --model ${model_flag} \
        --model_args pretrained=${model_name},revision=${revision},trust_remote_code=True${extra} \
        --tasks ${task_name} \
        --device cuda \
        --output_path results/${revision} \
        --batch_size auto:10 \
        --num_fewshot 0 \
        --log_samples \
        --include_path tasks/

    echo "Completed Global PIQA evaluation for ${lang}"
done
