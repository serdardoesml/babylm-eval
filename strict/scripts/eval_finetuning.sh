#!/bin/bash

set -euo pipefail

if [[ -f ../.env ]]; then
    set -a
    source ../.env
    set +a
fi

# Default values
MODEL_PATH=""
LR=3e-5
BSZ=32
BIG_BSZ=16
MAX_EPOCHS=10
WSC_EPOCHS=30
SEED=42
SEQUENCE_LENGTH=512
COMPILE_FLAG="--compile"
LORA_RANK=256
LORA_ALPHA=512
LORA_DROPOUT=0.0
LORA_TARGETS="qkv,out,up,down"
LORA_FLAG="--lora"
WANDB_FLAG=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --bsz)
            BSZ="$2"
            shift 2
            ;;
        --big_bsz)
            BIG_BSZ="$2"
            shift 2
            ;;
        --max_epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --wsc_epochs)
            WSC_EPOCHS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --sequence_length)
            SEQUENCE_LENGTH="$2"
            shift 2
            ;;
        --compile)
            COMPILE_FLAG="--compile"
            shift
            ;;
        --no_compile|--no-compile)
            COMPILE_FLAG="--no-compile"
            shift
            ;;
        --lora_rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --lora_alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --lora_dropout)
            LORA_DROPOUT="$2"
            shift 2
            ;;
        --lora_targets)
            LORA_TARGETS="$2"
            shift 2
            ;;
        --no_lora)
            LORA_FLAG="--no-lora"
            shift
            ;;
        --wandb)
            WANDB_FLAG="--wandb"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

for task in {boolq,multirc}; do
    echo $task

    python -m evaluation_pipeline.finetune.run \
        --model_name_or_path "$MODEL_PATH" \
        --train_data "evaluation_data/full_eval/glue_filtered/$task.train.jsonl" \
        --valid_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --predict_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --task "$task" \
        --num_labels 2 \
        --batch_size $BIG_BSZ \
        --learning_rate $LR \
        --num_epochs $MAX_EPOCHS \
        --sequence_length $SEQUENCE_LENGTH \
        --results_dir "results" \
        --save \
        --save_dir "models" \
        --metrics accuracy f1 mcc \
        --metric_for_valid accuracy \
        --seed $SEED \
        --verbose \
        --padding_side left \
        --take_final \
        $COMPILE_FLAG \
        $LORA_FLAG \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --lora_targets "$LORA_TARGETS" \
        $WANDB_FLAG
done

python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/glue_filtered/rte.train.jsonl" \
    --valid_data "evaluation_data/full_eval/glue_filtered/rte.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/glue_filtered/rte.valid.jsonl" \
    --task rte \
    --num_labels 2 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $MAX_EPOCHS \
    --sequence_length $SEQUENCE_LENGTH \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy f1 mcc \
    --metric_for_valid accuracy \
    --seed $SEED \
    --verbose \
    --padding_side left \
    --take_final \
    $COMPILE_FLAG \
    $LORA_FLAG \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_targets "$LORA_TARGETS" \
    $WANDB_FLAG

python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/glue_filtered/wsc.train.jsonl" \
    --valid_data "evaluation_data/full_eval/glue_filtered/wsc.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/glue_filtered/wsc.valid.jsonl" \
    --task wsc \
    --num_labels 2 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $WSC_EPOCHS \
    --sequence_length $SEQUENCE_LENGTH \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy f1 mcc \
    --metric_for_valid accuracy \
    --seed $SEED \
    --verbose \
    --padding_side left \
    --take_final \
    $COMPILE_FLAG \
    $LORA_FLAG \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_targets "$LORA_TARGETS" \
    $WANDB_FLAG

for task in {mrpc,qqp}; do
        
    python -m evaluation_pipeline.finetune.run \
        --model_name_or_path "$MODEL_PATH" \
        --train_data "evaluation_data/full_eval/glue_filtered/$task.train.jsonl" \
        --valid_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --predict_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --task "$task" \
        --num_labels 2 \
        --batch_size $BSZ \
        --learning_rate $LR \
        --num_epochs $MAX_EPOCHS \
        --sequence_length $SEQUENCE_LENGTH \
        --results_dir "results" \
        --save \
        --save_dir "models" \
        --metrics accuracy f1 mcc \
        --metric_for_valid f1 \
        --seed $SEED \
        --verbose \
	--padding_side left \
	--take_final \
        $COMPILE_FLAG \
        $LORA_FLAG \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --lora_targets "$LORA_TARGETS" \
        $WANDB_FLAG
done

python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/glue_filtered/mnli.train.jsonl" \
    --valid_data "evaluation_data/full_eval/glue_filtered/mnli.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/glue_filtered/mnli.valid.jsonl" \
    --task mnli \
    --num_labels 3 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $MAX_EPOCHS \
    --sequence_length $SEQUENCE_LENGTH \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy \
    --metric_for_valid accuracy \
    --seed $SEED \
    --verbose \
    --padding_side left \
    --take_final \
    $COMPILE_FLAG \
    $LORA_FLAG \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_targets "$LORA_TARGETS" \
    $WANDB_FLAG
