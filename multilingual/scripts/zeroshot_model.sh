#!/bin/bash
model_name=""
langs="eng nld zho"
revision="main"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name) model_name="$2"; shift 2 ;;
        --langs) langs="$2"; shift 2 ;;
        --revision) revision="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$model_name" ]]; then
    echo "Error: --model_name is required"; exit 1
fi

for lang in $langs; do
    task_name="zeroshot_${lang}"
    echo "Evaluating ${lang} (revision: ${revision})"
    python3 -m lm_eval \
        --model hf \
        --model_args pretrained=${model_name},revision=${revision},trust_remote_code=True \
        --tasks ${task_name} \
        --device cuda \
        --output_path ../results/${revision} \
        --batch_size auto:10 \
        --num_fewshot 0 \
        --log_samples \
        --include_path tasks/

    echo "Completed evaluation for ${lang}"
done
