#!/bin/bash
# MECO delta log-likelihood evaluation, mirroring scripts/zeroshot_model.sh.
model_name=""
langs="eng nld zho"
revision="main"
backend="auto"
device="cuda"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name) model_name="$2"; shift 2 ;;
        --langs)      langs="$2";      shift 2 ;;
        --revision)   revision="$2";   shift 2 ;;
        --backend)    backend="$2";    shift 2 ;;
        --device)     device="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$model_name" ]]; then
    echo "Error: --model_name is required"; exit 1
fi

# Run from multilingual/ so `python -m meco.meco_py.cli` resolves.
cd "$(dirname "$0")/.." || exit 1
python3 -m meco.meco_py.cli \
    --model_name "$model_name" \
    --langs "$langs" \
    --revision "$revision" \
    --backend "$backend" \
    --device "$device"
