#!/bin/bash
# Launch the (hidden) Hanzi minimal-pair tasks (hanzi_structure, hanzi_pinyin) for a model at
# a single revision. Hanzi is a Chinese-only task group, so this is a no-op unless `zho` is in
# --langs. Kept separate from zeroshot_model.sh (which also runs Hanzi inside the Chinese group)
# so it can be re-run on its own — mirroring the global_piqa_model.sh / meco_model.sh split.
#
# Its custom scorer (tasks/hanzi/utils.py) needs the evaluated model's tokenizer to flag any
# minimal pair containing an UNK token as incorrect; it reads the tokenizer from HANZI_TOKENIZER /
# HANZI_TOKENIZER_REVISION. The BOS handling is identical to global_piqa_model.sh: these BabyLM
# tokenizers auto-append both <s> and </s> on encode(); the in-repo `hf-bos` model (via
# scripts/run_lm_eval.py) tokenizes with add_special_tokens=False and prepends a single BOS,
# reproducing the strict harness. Opt out with --bos_fix 0 to use the stock `--model hf`.
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

# Hanzi is Chinese-only — skip cleanly when zho is not requested.
if [[ " $langs " != *" zho "* ]]; then
    echo "Hanzi — skipped (zho not in --langs: '${langs}')"
    exit 0
fi

# Hanzi's custom scorer uses the evaluated model's tokenizer to mark any minimal pair
# containing an UNK token as incorrect.
export HANZI_TOKENIZER="$model_name"
export HANZI_TOKENIZER_REVISION="$revision"

# Select the BOS-fix model (in-repo hf-bos) or the stock hf model.
if [[ "$bos_fix" == "1" ]]; then
    runner="python3 scripts/run_lm_eval.py"; model_flag="hf-bos"; extra=",add_bos_token=True"
else
    runner="python3 -m lm_eval"; model_flag="hf"; extra=""
fi

echo "Evaluating Hanzi (revision: ${revision})"
$runner \
    --model ${model_flag} \
    --model_args pretrained=${model_name},revision=${revision},trust_remote_code=True${extra} \
    --tasks hanzi_structure,hanzi_pinyin \
    --device cuda \
    --output_path results/${revision} \
    --batch_size auto:10 \
    --num_fewshot 0 \
    --log_samples \
    --include_path tasks/

echo "Completed Hanzi evaluation (revision: ${revision})"
