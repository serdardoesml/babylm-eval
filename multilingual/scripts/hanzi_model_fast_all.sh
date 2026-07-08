#!/bin/bash
# Drive hanzi_model.sh across all checkpoint revisions (chck_1M .. chck_1000M), mirroring
# global_piqa_model_fast_all.sh. Run from multilingual/ as `bash scripts/hanzi_model_fast_all.sh`
# so hanzi_model.sh's relative paths (results/, tasks/) resolve correctly. Hanzi is Chinese-only,
# so hanzi_model.sh self-skips when zho is not in --langs (this becomes a no-op for such models).
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
    echo "Evaluating Hanzi revision ${revision}"
    bash scripts/hanzi_model.sh --model_name "$model_name" --langs "$langs" --revision "$revision" --bos_fix "$bos_fix"
done

for i in {10..100..10}; do
    revision="chck_${i}M"
    echo "Evaluating Hanzi revision ${revision}"
    bash scripts/hanzi_model.sh --model_name "$model_name" --langs "$langs" --revision "$revision" --bos_fix "$bos_fix"
done

for i in {200..1000..100}; do
    revision="chck_${i}M"
    echo "Evaluating Hanzi revision ${revision}"
    bash scripts/hanzi_model.sh --model_name "$model_name" --langs "$langs" --revision "$revision" --bos_fix "$bos_fix"
done
