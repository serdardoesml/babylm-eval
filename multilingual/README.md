# Multilingual Track Evaluation

This directory contains the evaluation pipeline for the **multilingual track** of the 2026 BabyLM Challenge. It supports English (eng), Dutch (nld), and Chinese (zho), and is built on top of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (zero-shot) and a custom fine-tuning script (finetune tasks).

Incomplete evaluation is explicitly allowed: you may submit results for only the language(s) your model covers. Missing tasks are set to 0 and factored into the average scores on the leaderboard.

---

## Tasks

Evaluation is divided into **zero-shot** tasks (scored directly by the LM's log-probabilities) and **finetune** tasks (trained per-task with a small classification head).

### Zero-shot

| Language | Tasks |
|----------|-------|
| English  | BLiMP, HellaSwag, MultiBLiMP, Winogrande, XStoryCloze, GlobalPIQA |
| Dutch    | BLiMP-NL, HellaSwag, MultiBLiMP, Winogrande, XCOMPS, XStoryCloze, GlobalPIQA |
| Chinese  | HellaSwag, Winogrande, XCOMPS, XStoryCloze, ZhoBLiMP, Hanzi Structure, Hanzi Pinyin, GlobalPIQA |

Custom task definitions are in `tasks/`.

The Chinese group includes the
[Hanzi Structure](https://huggingface.co/datasets/chinese-babylm-org/hanzi-structure)
and [Hanzi Pinyin](https://huggingface.co/datasets/chinese-babylm-org/hanzi-pinyin)
minimal-pair tasks. Their scorer uses the evaluated model's tokenizer and
marks an item incorrect if either sentence contains an unknown token. The
standard zero-shot scripts configure this automatically.

### Finetune

| Language | Tasks |
|----------|-------|
| English  | ARC, Belebele, BMLama, MNLI, SIB-200, TruthfulQA, XNLI |
| Dutch    | ARC, Belebele, BMLama, INCLUDE, MNLI, SIB-200, TruthfulQA |
| Chinese  | ARC, Belebele, BMLama, INCLUDE, MNLI, SIB-200, TruthfulQA, XNLI |

---

## Running Evaluation

All scripts are run **from this `multilingual/` directory** (e.g. `bash scripts/...`).

### Quick start (recommended)

Two wrapper scripts run every required eval for a model in one command:

```bash
# Final-checkpoint eval: zero-shot + Global PIQA + MECO + finetune
bash scripts/eval_model_full.sh --model_name YOUR_MODEL

# Intermediate-checkpoint learning curves: zero-shot (including Hanzi) +
# Global PIQA + MECO across chck_1M .. chck_1000M
bash scripts/eval_model_fast.sh --model_name YOUR_MODEL
```

Both accept `--langs "eng nld"` to restrict languages and `--bos_fix 0` to disable the
[BOS tokenization fix](#bos-tokenization-fix); `eval_model_full.sh` also accepts `--revision`.
Finetune is **not** part of the fast eval and is scored once at the final
checkpoint by `eval_model_full.sh`. Hanzi and MECO run at each checkpoint and
are scored server-side for the final checkpoint. When both are done, [collate](#collating-results-for-submission)
with `--fast`.

To (re-)run **only the tasks newly added for 2026** — Hanzi, Global PIQA, MECO, and the
finetune POS task — without redoing the full eval:

```bash
bash scripts/eval_hidden_tasks.sh --model_name YOUR_MODEL
```

The sections below document the individual sub-scripts, which you can also run directly.

### Zero-shot

Evaluate a model across all three languages:

```bash
bash scripts/zeroshot_model.sh --model_name YOUR_MODEL
```

To restrict to specific languages:

```bash
bash scripts/zeroshot_model.sh --model_name YOUR_MODEL --langs "eng nld"
```

Results are written to `results/main/<org__model>/results_<timestamp>.json`.

### Finetune

```bash
bash scripts/finetune_model.sh --model_name YOUR_MODEL --langs "eng nld zho"
```

Optional hyperparameter flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--lr` | `5e-5` | Learning rate |
| `--bsz` | `64` | Batch size |
| `--max_epochs` | `10` | Max training epochs |
| `--patience` | `3` | Early stopping patience |
| `--seed` | `12` | Random seed |

Results are written to `finetune/results/<model>/<lang>/<task>/`.

**POS tagging is cross-lingual.** Unlike the other finetune tasks (one model per
language), POS trains a *single* model on a uniform mixture of all requested
languages' UD treebanks — each language contributes an equal number of sentences
— then evaluates it per language. Its results go to
`finetune/results/<model>/pos/<lang>/`, and collate reports them as `pos: {en, nl, zh}`.
To finetune POS on a subset, pass those languages (e.g. `--langs "eng nld"`); a
single language falls back to a monolingual run.

### Global PIQA

Global PIQA is kept in its own script because it is scored with `acc_norm`
(length-normalized) only — never plain `acc` — so its numbers are never mixed
with the acc-based zero-shot tasks.

```bash
bash scripts/global_piqa_model.sh --model_name YOUR_MODEL --langs "eng nld zho"
```

Results are written alongside the zero-shot results under `results/<revision>/<org__model>/`.

### MECO

MECO measures how well the model's per-word **surprisal** predicts human reading
times (a delta log-likelihood). It is a zero-shot / probing eval — it does **not**
finetune the model — but it runs through `minicons` rather than the lm-eval harness,
so it has its own script (now in `scripts/`, alongside the others):

```bash
bash scripts/meco_model.sh --model_name YOUR_MODEL --langs "eng nld zho"
# or directly:
python -m meco.meco_py.cli --model_name YOUR_MODEL --langs "eng nld zho"
```

Per requested language it computes native reading (`meco_l1`) and, when English is
in the mixture, English-as-L2 reading (`meco_l2`). Raw per-word surprisals are written
to `meco/results/<revision>/meco_<org__model>/predictions_meco.json`; final scores are
computed **server-side** from that file. See `meco/README.md` for options.

### Intermediate Checkpoint Evaluations

As in the Strict and Strict-small tracks, we expect challenge participants to evaluate intermediate checkpoints on zero-shot tasks. These can be run via:

```bash
bash scripts/zeroshot_model_fast_all.sh --model_name YOUR_MODEL # to run on all languages
bash scripts/zeroshot_model_fast_all.sh --model_name YOUR_MODEL --langs "eng nld" # to restrict to specific languages
```

The result for each revision will be written to `results/<revision_name>/<org__model>/results_<timestamp>.json`. The scripts assume that revisions are named in the form `chck_1M, chck_2M, ..., chck_1000M`. If your training scripts follows a different logic, you should alter the script to match the revision names.

### BOS Tokenization Fix

We found a bug in how the BabyLM baselines' tokenizers interact with the lm-eval repo. lm-eval's loglikelihood / multiple-choice path tokenizes the context (prompt) and the whole (prompt + continuation) separately, then recovers the continuation tokens via `whole[len(context):]`. This assumes the tokenizer adds the same leading special tokens to both strings and nothing in the middle or at the end, which is not the case for the BabyLM baselines that wrap strings as `<s> ... </s>`. 

We patch this with an in-repo lm-eval model, `hf-bos` (`scripts/bos_hf_model.py`), that tokenizes with `add_special_tokens=False` (dropping the spurious `</s>`) and prepends exactly one BOS `<s>` when `add_bos_token=True`. This is the default option for both `zeroshot_model.sh` and `zeroshot_model_fast_all.sh`. This fix will only cause problems if your model was trained without any BOS tokens whatsoever, where adding a BOS token will push the model out-of-distribution. In this case, you can run evaluations without the fix by passing `--bos_fix 0` as below:

```bash
bash scripts/zeroshot_model_fast_all.sh --model_name YOUR_MODEL --bos_fix 0
```

---

## Collating Results for Submission

Once evaluation is complete, run `scripts/collate_results.py` to produce the submission files:

```bash
python scripts/collate_results.py --model_name YOUR_MODEL --fast
```

This produces **two output files**:

### `<model_name>_submission.json` — standard-task scores file

This is one of the two files you upload to the leaderboard. It contains
pre-computed scores for the public standard tasks. Hanzi and MECO are omitted:
the leaderboard computes those scores server-side from the raw prediction file.

```json
{
  "blimp":              {"blimp": 0.734},
  "hellaswag_en_mubench": {"hellaswag_en_mubench": 0.265},
  "arc":                {"en": 0.248, "nl": 0.244}
}
```

Zeroshot tasks use the task name as their single key. Finetune tasks use the language code as the key so results from multiple languages are grouped under one benchmark name.

### `<model_name>_predictions.json` — raw predictions file

This file is also uploaded to the leaderboard alongside the scores file. It is
used to score the hidden Hanzi and MECO tasks server-side, and to verify other
tasks. It has these top-level keys:

- **`"zeroshot"`** — the raw lm-eval `results` dict merged across all `results_*.json` files (includes individual subtask rows such as BLiMP paradigms)
- **`"finetune"`** — keyed as `"{task}_{lang}"` (e.g. `"arc_nl"`), each holding the list of per-example predictions from `predictions.txt`
- **`"hidden"`** — raw Hanzi candidate log-likelihoods/UNK flags and MECO
  per-word surprisals. Final scores for these tasks are not accepted from the
  participant-supplied score file.
- **`"fast_eval_results"` - a list of result dictionaries in the same format as the dictionary held in `<model_name>_submission.json`. We assume that the list is ordered in terms of checkpoint word count.
- **`"fast_eval_hidden"`** — a list aligned with the fast checkpoints. Each
  entry contains an explicit `"revision"` and a `"hidden"` dictionary with
  raw Hanzi and MECO predictions for server-side scoring.

```json
{
  "zeroshot": {
    "blimp": {"acc,none": 0.734, ...},
    "blimp_adjunct_island": {"acc,none": 0.812, ...}
  },
  "finetune": {
    "arc_en": [{"index": 0, "prediction": "C"}, ...],
    "arc_nl": [{"index": 0, "prediction": "B"}, ...]
  },
  "hidden": {
    "hanzi_structure": {
      "0:<doc_hash>": {"scores": [-12.3, -15.8], "has_unk": false}
    },
    "meco_l1": {
      "zho": {"1:1": 8.42, "1:2": 11.05}
    }
  },
  "fast_eval_results": [
     {"blimp": ...}, {"blimp": ...}, ...
  ],
  "fast_eval_hidden": [
    {
      "revision": "chck_1M",
      "hidden": {
        "hanzi_structure": {
          "0:<doc_hash>": {"scores": [-12.3, -15.8], "has_unk": false}
        },
        "meco_l1": {
          "zho": {"1:1": 8.42, "1:2": 11.05}
        }
      }
    }
  ]
}
```

Note that the collation script assumes the intermediate checkpoint names follow the format of `chck_{word_count}M`. If this does not follow your naming conventions during training, you should change the `REVISIONS` list.

### Missing task warnings

Before writing anything, the script checks whether the submission is complete for each submitted language. If a language has at least one result but is missing other expected tasks, a warning is printed for each:

```
Warning: zeroshot task 'multiblimp_eng' (eng) is missing — will be scored as 0.
Warning: finetune task 'xnli' (en) is missing — will be scored as 0.
```

Languages with no results at all are silently skipped (intentional partial submissions are fine). If everything is present you will see:

```
All tasks present for every submitted language.
```

### Custom output paths

```bash
python scripts/collate_results.py \
    --model_name YOUR_MODEL \
    --output path/to/submission.json \
    --output_predictions path/to/predictions.json
```

---

## Viewing Results Locally

Two helper scripts let you inspect results without uploading anything.

**Zero-shot results table** (markdown, grouped by language):

```bash
python scripts/print_results_table.py
```

**Finetune results table** (markdown, grouped by language):

```bash
python scripts/print_finetune_results.py
```

---

## Submitting to the Leaderboard

Upload both output files from `collate_results.py` on the leaderboard submission page:

- **Results file** (`*_submission.json`) — supplies public standard-task scores
- **Predictions file** (`*_predictions.json`) — required for the multilingual
  track; Hanzi and MECO are scored from this file on the leaderboard

The leaderboard is live at: [![Leaderboard](https://img.shields.io/badge/🤗-Leaderboard-yellow)](https://huggingface.co/spaces/BabyLM-community/BabyLM-Leaderboard-2026)
