#!/bin/bash
# Drive meco_model.sh across all checkpoint revisions (chck_1M .. chck_1000M), mirroring
# hanzi_model_fast_all.sh / global_piqa_model_fast_all.sh. This is the standalone MECO
# equivalent of the [3/3] MECO loop inside scripts/eval_model_fast.sh, so it can be re-run
# on its own. Run from multilingual/ as `bash scripts/meco_model_fast_all.sh` so
# meco_model.sh's relative paths (meco/, results/) resolve correctly.
model_name=""
langs="eng nld zho"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name) model_name="$2"; shift 2 ;;
        --langs) langs="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$model_name" ]]; then
    echo "Error: --model_name is required"; exit 1
fi

for i in {1..9} {10..100..10} {200..1000..100}; do
    revision="chck_${i}M"
    echo "Evaluating MECO revision ${revision}"
    bash scripts/meco_model.sh --model_name "$model_name" --langs "$langs" --revision "$revision"
done
