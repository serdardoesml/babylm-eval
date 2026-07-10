#!/usr/bin/env python
# coding=utf-8
"""Finetuning for per-token classification (POS tagging) on Universal Dependencies.

This mirrors `finetune_classification.py` but predicts one label *per word*
instead of one label per sentence. Shared overhead (arguments, logging,
checkpoint detection, config/tokenizer loading, early stopping) lives in
`finetune_common.py`; the data comes from `pos_data.build_pos_mixture`.

Subword handling: a word is usually split into several subword tokens. We run
the encoder over the subwords, then **mean-pool** the subword hidden states
belonging to each word into a single word vector before the classifier head, so
the model emits exactly one prediction per gold word-level label.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import datasets
import transformers
from datasets import DatasetDict
from sklearn.metrics import f1_score
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    EarlyStoppingCallback,
    HfArgumentParser,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

# Reuse the model/config/tokenizer argument definition from the sequence-
# classification script so the two stay in sync; that file is left untouched.
from finetune_classification import ModelArguments
from pos_data import DEFAULT_UD_CONFIGS, build_pos_mixture

require_version("datasets>=1.8.0")

logger = logging.getLogger(__name__)


# --- Shared boilerplate, kept local so finetune_classification.py is untouched ---

def setup_logging(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


def detect_last_checkpoint(training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")
    return last_checkpoint


def build_early_stopping(patience, training_args):
    if not patience:
        return None
    training_args.save_total_limit = 1
    training_args.load_best_model_at_end = True
    training_args.eval_strategy = "epoch"
    return [EarlyStoppingCallback(early_stopping_patience=patience, early_stopping_threshold=0.001)]


def save_metrics_json(directory, filename, metrics):
    """Write a metrics dict to <directory>/<filename> (creating the directory)."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), "w") as f:
        json.dump(metrics, f, indent=4, sort_keys=True)


def write_pos_predictions(directory, pred_ids, gold, label_list):
    """Write per-word predicted tags to <directory>/predictions.txt."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "predictions.txt"), "w") as writer:
        writer.write("index\tpredictions\n")
        for index, (row_pred, row_gold) in enumerate(zip(pred_ids, gold)):
            tags = [label_list[p] for p, g in zip(row_pred, row_gold) if g != -100]
            writer.write(f"{index}\t{' '.join(tags)}\n")


@dataclass
class DataTrainingArguments:
    """Arguments controlling the POS data we load and how we preprocess it."""

    language: Optional[str] = field(
        default=None,
        metadata={"help": "One or more space-separated language codes (e.g. 'en' or 'en nl zh'). Each picks "
                          "its default UD treebank unless --ud_config is given. Multiple languages are trained "
                          "jointly on a uniform cross-lingual mixture."},
    )
    ud_config: Optional[str] = field(
        default=None,
        metadata={"help": "Universal Dependencies treebank config, e.g. en_ewt. Overrides the per-language default."},
    )
    max_sentences: Optional[int] = field(
        default=4000,
        metadata={"help": "Subsample the pooled treebank to this many sentences before the 80/10/10 split."},
    )
    split_seed: int = field(
        default=42, metadata={"help": "Seed for the UD pooling/shuffle/split (independent of the training seed)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "Max total input length in subword tokens after tokenization; longer sequences are truncated."},
    )
    pad_to_multiple_of: Optional[int] = field(
        default=None, metadata={"help": "If set, pad sequence length up to a multiple of this (useful for fp16)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Truncate the number of training examples for debugging/quicker training."}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "Truncate the number of evaluation examples for debugging/quicker training."}
    )
    max_predict_samples: Optional[int] = field(
        default=None, metadata={"help": "Truncate the number of prediction examples for debugging/quicker training."}
    )
    patience: Optional[int] = field(
        default=None,
        metadata={"help": "Early-stopping patience in epochs (no improvement > epsilon before stopping)."},
    )

    def __post_init__(self):
        # Resolve --language / --ud_config into an ordered {label: ud_config} map.
        # One entry -> monolingual (unchanged); several -> cross-lingual mixture.
        if self.ud_config is not None:
            langs = self.language.split() if self.language else []
            if len(langs) > 1:
                raise ValueError("--ud_config can only be combined with a single --language.")
            label = langs[0][:2].lower() if langs else self.ud_config
            self.ud_configs = {label: self.ud_config}
            return
        if self.language is None:
            raise ValueError("Provide either --ud_config or --language.")
        configs: dict[str, str] = {}
        for raw in self.language.split():
            lang = raw[:2].lower()
            if lang not in DEFAULT_UD_CONFIGS:
                raise ValueError(
                    f"No default UD config for language '{raw}'. "
                    f"Known: {sorted(DEFAULT_UD_CONFIGS)}. Pass --ud_config explicitly."
                )
            configs[lang] = DEFAULT_UD_CONFIGS[lang]
        if not configs:
            raise ValueError("--language must name at least one language.")
        self.ud_configs = configs


def pool_subwords(hidden, word_ids, max_words):
    """Mean-pool subword hidden states into per-word representations.

    Args:
        hidden: [batch, seq, hidden] encoder outputs.
        word_ids: [batch, seq] long tensor; entry w means "subword belongs to
            word w", and -1 marks specials / padding (ignored in the mean).
        max_words: number of word slots in the output (the label width).

    Returns:
        [batch, max_words, hidden]. Word slots with no subwords (padding words)
        come out as zero vectors; their labels are -100 so they never count.
    """
    valid = (word_ids >= 0)
    idx = word_ids.clamp(min=0)
    onehot = torch.zeros(hidden.size(0), hidden.size(1), max_words, device=hidden.device, dtype=hidden.dtype)
    onehot.scatter_(2, idx.unsqueeze(-1), 1.0)
    onehot = onehot * valid.unsqueeze(-1)                 # zero out specials/padding
    counts = onehot.sum(dim=1).clamp(min=1.0)             # [batch, max_words]
    pooling = onehot.transpose(1, 2)                      # [batch, max_words, seq]
    word_reps = torch.bmm(pooling, hidden) / counts.unsqueeze(-1)
    return word_reps


class PooledTokenClassifier(PreTrainedModel):
    """Base encoder + mean subword pooling + linear per-word classifier head."""

    def __init__(self, model_args, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.encoder = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            trust_remote_code=True,
        )
        dropout = getattr(config, "classifier_dropout", None)
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, word_ids=None, labels=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        max_words = labels.size(1) if labels is not None else int(word_ids.max().item()) + 1
        word_reps = pool_subwords(hidden, word_ids, max_words)
        logits = self.classifier(self.dropout(word_reps))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits)


@dataclass
class DataCollatorForSubwordPooling:
    """Pads subword fields to the batch's max sequence length and word-level
    labels to the batch's max word count (-100 for padding words)."""

    pad_token_id: int
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        max_words = max(len(f["labels"]) for f in features)

        input_ids, attention_mask, word_ids, labels = [], [], [], []
        for f in features:
            seq_pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * seq_pad)
            attention_mask.append(f["attention_mask"] + [0] * seq_pad)
            word_ids.append(f["word_ids"] + [-1] * seq_pad)
            word_pad = max_words - len(f["labels"])
            labels.append(f["labels"] + [self.label_pad_token_id] * word_pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "word_ids": torch.tensor(word_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    callbacks = build_early_stopping(data_args.patience, training_args)

    setup_logging(training_args)
    last_checkpoint = detect_last_checkpoint(training_args)
    set_seed(training_args.seed)

    # --- Data: build the 80/10/10 POS split(s). A single language is one
    # treebank; several languages are trained jointly on a uniform mixture. ---
    raw_datasets: DatasetDict = build_pos_mixture(
        data_args.ud_configs,
        max_sentences=data_args.max_sentences,
        seed=data_args.split_seed,
    )
    logger.info(f"Loaded UD config(s) {data_args.ud_configs}: "
                f"{ {k: len(v) for k, v in raw_datasets.items()} }")

    # `upos` is a Sequence(ClassLabel); pull the tag names (ids are already 0..N-1).
    upos_feature = raw_datasets["train"].features["upos"]
    label_list = upos_feature.feature.names
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="pos",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        trust_remote_code=True,
    )
    tokenizer_kwargs = dict(
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        trust_remote_code=True,
    )
    tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    # Byte-level BPE tokenizers (GPT-2/RoBERTa families) require add_prefix_space
    # to accept pre-split words via is_split_into_words; harmless for others.
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True, **tokenizer_kwargs)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for word_ids()-based subword pooling.")

    model = PooledTokenClassifier(model_args, config)
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        pad_token_id = tokenizer.pad_token_id
    else:
        model.config.pad_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.eos_token_id

    if model_args.freeze_model:
        # Freeze the base encoder; keep only the pooling classifier head trainable.
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {i: l for i, l in enumerate(label_list)}

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        tokenized = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        batch_word_ids, batch_labels = [], []
        for i, upos in enumerate(examples["upos"]):
            word_ids = tokenized.word_ids(batch_index=i)
            # Words that survive truncation, in order of first appearance.
            ordered = []
            for wid in word_ids:
                if wid is not None and (not ordered or wid != ordered[-1]):
                    ordered.append(wid)
            remap = {wid: new for new, wid in enumerate(ordered)}
            batch_word_ids.append([remap[wid] if wid is not None else -1 for wid in word_ids])
            batch_labels.append([upos[wid] for wid in ordered])
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "word_ids": batch_word_ids,
            "labels": batch_labels,
        }

    with training_args.main_process_first(desc="dataset map pre-processing"):
        processed = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenizing and aligning POS labels",
        )

    def take(split, n):
        ds = processed[split]
        if n is not None:
            ds = ds.select(range(min(len(ds), n)))
        return ds

    # Languages resolved from --language; >1 means a joint cross-lingual model
    # that we score per language on its own validation_<lang>/test_<lang> split.
    langs = list(data_args.ud_configs)
    multi = len(langs) > 1

    train_dataset = take("train", data_args.max_train_samples) if training_args.do_train else None
    # The Trainer always evaluates the pooled validation set (for early stopping).
    eval_dataset = take("validation", data_args.max_eval_samples) if training_args.do_eval else None
    # Single-language prediction uses the pooled test split; the multi-language
    # path predicts per language (test_<lang>) below.
    predict_dataset = take("test", data_args.max_predict_samples) if (training_args.do_predict and not multi) else None

    data_collator = DataCollatorForSubwordPooling(
        pad_token_id=pad_token_id, pad_to_multiple_of=data_args.pad_to_multiple_of
    )

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return torch.argmax(logits, dim=-1), labels

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        labels = p.label_ids
        mask = labels != -100
        preds, labels = preds[mask], labels[mask]
        return {
            "accuracy": (preds == labels).astype(np.float32).mean().item(),
            "macro_f1": f1_score(y_true=labels, y_pred=preds, average="macro"),
        }

    training_args.metric_for_best_model = "accuracy"
    training_args.greater_is_better = True
    training_args.save_strategy = "no"

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if multi:
            # Score the joint model on each language's own dev split, writing a
            # separate eval_results.json per language under <output_dir>/<lang>/.
            for lang in langs:
                ds = take(f"validation_{lang}", data_args.max_eval_samples)
                metrics = trainer.evaluate(eval_dataset=ds, metric_key_prefix="eval")
                metrics["eval_samples"] = len(ds)
                save_metrics_json(os.path.join(training_args.output_dir, lang), "eval_results.json", metrics)
                logger.info(f"[{lang}] eval_accuracy = {metrics.get('eval_accuracy')}")
        else:
            metrics = trainer.evaluate(eval_dataset=eval_dataset)
            metrics["eval_samples"] = len(eval_dataset)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        targets = ([(lang, take(f"test_{lang}", data_args.max_predict_samples)) for lang in langs]
                   if multi else [(None, predict_dataset)])
        for lang, ds in targets:
            output = trainer.predict(ds, metric_key_prefix="predict")
            pred_ids = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
            out_dir = os.path.join(training_args.output_dir, lang) if lang else training_args.output_dir
            write_pos_predictions(out_dir, pred_ids, output.label_ids, label_list)
            if lang:
                save_metrics_json(out_dir, "predict_results.json", output.metrics)
                logger.info(f"[{lang}] predict_accuracy = {output.metrics.get('predict_accuracy')}")
            else:
                trainer.log_metrics("predict", output.metrics)
                trainer.save_metrics("predict", output.metrics)


if __name__ == "__main__":
    main()
