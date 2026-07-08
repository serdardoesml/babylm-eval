#!/bin/bash
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

# Hanzi's custom scorer uses the evaluated model's tokenizer to mark any
# minimal pair containing an UNK token as incorrect.
export HANZI_TOKENIZER="$model_name"
export HANZI_TOKENIZER_REVISION="$revision"

# Select the BOS-fix model (in-repo hf-bos) or the stock hf model.
if [[ "$bos_fix" == "1" ]]; then
    runner="python3 scripts/run_lm_eval.py"; model_flag="hf-bos"; extra=",add_bos_token=True"
else
    runner="python3 -m lm_eval"; model_flag="hf"; extra=""
fi

for lang in $langs; do
    task_name="zeroshot_${lang}"
    echo "Evaluating ${lang} (revision: ${revision})"
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

    echo "Completed evaluation for ${lang}"
done
