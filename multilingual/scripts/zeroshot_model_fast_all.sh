#!/bin/bash
model_name=""
langs="eng nld zho"
bos_fix=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name) model_name="$2"; shift 2 ;;
        --langs) langs="$2"; shift 2 ;;
        --bos_fix) bos_fix="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$model_name" ]]; then
    echo "Error: --model_name is required"; exit 1
fi

for i in {1..9}; do
    revision="chck_${i}M"
    echo "Evaluating revision ${revision}"
    bash scripts/zeroshot_model.sh --model_name "$model_name" --langs "$langs" --revision "$revision" --bos_fix "$bos_fix"
done

for i in {10..100..10}; do
    revision="chck_${i}M"
    echo "Evaluating revision ${revision}"
    bash scripts/zeroshot_model.sh --model_name "$model_name" --langs "$langs" --revision "$revision" --bos_fix "$bos_fix"
done

for i in {200..1000..100}; do
    revision="chck_${i}M"
    echo "Evaluating revision ${revision}"
    bash scripts/zeroshot_model.sh --model_name "$model_name" --langs "$langs" --revision "$revision" --bos_fix "$bos_fix"
done
