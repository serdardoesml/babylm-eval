#!/usr/bin/env python3
"""Collate zeroshot and finetune multilingual results into a single submission JSON.

Output format: {task_name: {subtask_key: accuracy}} — compatible with the leaderboard validator.
Zeroshot tasks use the task name as their single subtask key.
Finetune tasks use the language code as the subtask key (so cross-lingual results are averaged).

Usage:
    python scripts/collate_results.py --model_name gpt2-baseline-BabyLM-2026-Strict
"""

import argparse
import json
import math
from pathlib import Path

# These tasks are scored by the leaderboard from raw predictions. Their
# locally-computed metrics must never be copied into the score submission.
SERVER_SCORED_TASKS = {"hanzi_structure", "hanzi_pinyin", "meco_l1", "meco_l2"}

# Expected tasks per language — used to warn about partial submissions.
# Keys match the task names as they appear in the collated submission JSON.
EXPECTED_ZEROSHOT = {
    "eng": {"blimp", "hellaswag_en_mubench", "multiblimp_eng", "winogrande_en_mubench", "xstorycloze_en_mubench",
            "global_piqa_parallel_en", "global_piqa_nonparallel_en"},
    "nld": {"blimp_nl", "hellaswag_nl_mubench", "multiblimp_nld", "winogrande_nl_mubench", "xcomps_nl", "xstorycloze_nl_mubench",
            "global_piqa_parallel_nl", "global_piqa_nonparallel_nl"},
    "zho": {"hellaswag_zh_mubench", "winogrande_zh_mubench", "xcomps_zh", "xstorycloze_zh_mubench", "zhoblimp",
            "hanzi_structure", "hanzi_pinyin",
            "global_piqa_parallel_zh", "global_piqa_nonparallel_zh"},
}

EXPECTED_FINETUNE = {
    "en": {"arc", "belebele", "bmlama", "mnli", "pos", "sib200", "truthfulqa", "xnli"},
    "nl": {"arc", "belebele", "bmlama", "include", "mnli", "pos", "sib200", "truthfulqa"},
    "zh": {"arc", "belebele", "bmlama", "include", "mnli", "pos", "sib200", "truthfulqa", "xnli"},
}

# Fast-eval checkpoint revisions (chck_1M .. chck_1000M). Hanzi and MECO are
# scored locally at each checkpoint and collated as raw numbers, so no per-revision
# predictions are uploaded.
REVISIONS = (
    [f"chck_{i}M" for i in range(1, 10)]
    + [f"chck_{i * 10}M" for i in range(1, 10)]
    + [f"chck_{i * 100}M" for i in range(1, 11)]
)


def warn_missing_zeroshot(
    zeroshot: dict[str, dict[str, float]],
    label: str,
    extra_present=(),
) -> None:
    """Warn about missing zeroshot tasks within any language that has at least one task present.

    A language is considered submitted when at least one of its tasks is present.
    Entirely absent languages are silently skipped — incomplete submissions are allowed.
    """
    warned = False
    for lang, expected in EXPECTED_ZEROSHOT.items():
        found = expected & (zeroshot.keys() | set(extra_present))
        if not found:
            continue
        for task in sorted(expected - found):
            print(f"[{label}] Warning: zeroshot task '{task}' ({lang}) is missing — will be scored as 0.")
            warned = True
    if not warned and zeroshot:
        print(f"[{label}] All zeroshot tasks present for every submitted language.")


def warn_missing_finetune(finetune: dict[str, dict[str, float]]) -> None:
    """Warn about missing finetune tasks within any language that has at least one task present."""
    warned = False
    for lang_code, expected in EXPECTED_FINETUNE.items():
        found_benchmarks = {task for task, subtasks in finetune.items() if lang_code in subtasks}
        if not found_benchmarks:
            continue
        for task in sorted(expected - found_benchmarks):
            print(f"[main] Warning: finetune task '{task}' ({lang_code}) is missing — will be scored as 0.")
            warned = True
    if not warned and finetune:
        print("[main] All finetune tasks present for every submitted language.")


def parse_zeroshot(results: dict, include_server_scored: bool = False) -> dict[str, dict[str, float]]:
    """Extract indent-1 tasks from a lm-eval results dict.

    Indentation in the alias field encodes depth:
      indent 0 → language group aggregate (e.g. zeroshot_eng)  → skip
      indent 1 → task row (e.g. blimp, hellaswag_en_mubench)   → keep
      indent 2 → subtask row (e.g. blimp_adjunct_island)        → skip (already folded into indent-1)

    Server-scored tasks (Hanzi) are dropped from the main submission — their
    authoritative scores come from raw predictions, so their locally-computed
    accuracy must never leak in. For fast/intermediate checkpoints, which are
    scored entirely locally, pass ``include_server_scored=True`` to keep Hanzi's
    lm-eval accuracy (which equals the leaderboard's ``_score_hanzi``).
    """
    out: dict[str, dict[str, float]] = {}
    expected_tasks = set().union(*EXPECTED_ZEROSHOT.values())
    for task_name, task_data in results.items():
        alias = task_data.get("alias", task_name)
        indent = len(alias) - len(alias.lstrip())
        # Global PIQA is scored with acc_norm (length-normalized) only — matching the
        # strict pipeline — so it never reports plain acc. Every other task uses acc.
        metric = "acc_norm,none" if task_name.startswith("global_piqa") else "acc,none"
        score = task_data.get(metric)
        # Standard language-group runs place benchmark tasks at indent 1.
        # Also accept a known benchmark at indent 0 so targeted task-only
        # reruns can be collated without pulling in nested benchmark subtasks.
        if (
            (include_server_scored or task_name not in SERVER_SCORED_TASKS)
            and (indent == 1 or task_name in expected_tasks)
            and score is not None
        ):
            out[task_name] = {task_name: score}
    return out


def load_zeroshot(
    results_dir: Path, model_name: str, include_server_scored: bool = False
) -> dict[str, dict[str, float]]:
    """Load all results_*.json files from the model's folder in results/.

    Folder name can be org__model or just model; matches on the part after the last '__'.
    Multiple JSON files in the same folder (one per language run) are merged.
    """
    if not results_dir.exists():
        print(f"Warning: no zeroshot results folder found at {results_dir}")
        return {}

    matches = [d for d in results_dir.iterdir() if d.is_dir() and d.name.split("__")[-1] == model_name]
    if not matches:
        print(f"Warning: no zeroshot results folder found for '{model_name}' in {results_dir}")
        return {}

    combined: dict[str, dict[str, float]] = {}
    for folder in matches:
        json_files = sorted(folder.glob("results_*.json"))
        if not json_files:
            print(f"Warning: no results_*.json found in {folder}")
            continue
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
            combined.update(parse_zeroshot(data["results"], include_server_scored))

    return combined


def _finetune_lang_task(result_file: Path) -> tuple[str, str]:
    """Map a finetune result path to (lang, task).

    Per-language tasks live at {model}/{lang}/{task}/. The joint cross-lingual
    POS model instead writes {model}/pos/{lang}/ (one model, scored per language),
    so a leading 'pos' component means the two path parts are swapped.
    """
    parent, child = result_file.parts[-3], result_file.parts[-2]
    if parent == "pos":
        return child, "pos"
    return parent, child


def load_finetune(finetune_dir: Path, model_name: str) -> dict[str, dict[str, float]]:
    """Load eval_results.json files from finetune/results/{model_name}/{lang}/{task}/.

    Also handles the joint cross-lingual POS layout {model_name}/pos/{lang}/.
    Returns {task: {lang: eval_accuracy}}.
    """
    model_dir = finetune_dir / model_name
    if not model_dir.exists():
        print(f"Warning: no finetune results folder found for '{model_name}' in {finetune_dir}")
        return {}

    out: dict[str, dict[str, float]] = {}
    for result_file in sorted(model_dir.glob("*/*/eval_results.json")):
        lang, task = _finetune_lang_task(result_file)
        with open(result_file) as f:
            data = json.load(f)
        acc = data.get("eval_accuracy")
        if acc is None:
            continue
        out.setdefault(task, {})[lang] = acc

    return out


def load_zeroshot_raw(results_dir: Path, model_name: str) -> dict:
    """Merge the raw lm-eval 'results' dicts from all results_*.json files for a model."""
    if not results_dir.exists():
        return {}

    matches = [d for d in results_dir.iterdir() if d.is_dir() and d.name.split("__")[-1] == model_name]
    if not matches:
        return {}

    merged: dict = {}
    for folder in matches:
        for json_file in sorted(folder.glob("results_*.json")):
            with open(json_file) as f:
                data = json.load(f)
            merged.update(data.get("results", {}))
    return merged


def load_finetune_predictions(finetune_dir: Path, model_name: str) -> dict[str, list[dict]]:
    """Load predictions.txt files from finetune/results/{model_name}/{lang}/{task}/.

    Returns {"{task}_{lang}": [{"index": int, "prediction": str}, ...]}.
    """
    model_dir = finetune_dir / model_name
    if not model_dir.exists():
        return {}

    out: dict[str, list[dict]] = {}
    for pred_file in sorted(model_dir.glob("*/*/predictions.txt")):
        lang, task = _finetune_lang_task(pred_file)
        key = f"{task}_{lang}"
        rows = []
        with open(pred_file) as f:
            header = f.readline()  # skip "index\tprediction" header
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t", 1)
                rows.append({"index": int(parts[0]), "prediction": parts[1] if len(parts) > 1 else ""})
        out[key] = rows
    return out


def _hanzi_sample_prediction(sample: dict) -> dict:
    """Reduce an lm-eval Hanzi sample to the raw fields needed server-side."""
    try:
        scores = [float(choice[0]) for choice in sample["filtered_resps"]]
        target = int(sample["target"])
        local_acc = float(sample["acc"])
        doc_hash = sample["doc_hash"]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("Malformed Hanzi lm-eval sample") from exc
    if len(scores) != 2 or not all(math.isfinite(score) for score in scores):
        raise ValueError(f"Invalid candidate scores for Hanzi item {doc_hash}")
    prediction = max(range(len(scores)), key=scores.__getitem__)
    # The custom lm-eval scorer forces an otherwise-correct item to zero when
    # either sentence contains the tokenizer's UNK token. Recover that raw
    # eligibility signal without copying the final accuracy into the upload.
    has_unk = prediction == target and local_acc == 0.0
    return {"scores": scores, "has_unk": has_unk}


def load_hanzi_predictions(results_dir: Path, model_name: str) -> dict[str, dict]:
    """Load the newest per-item lm-eval sample log for each Hanzi task."""
    if not results_dir.exists():
        return {}
    matches = [d for d in results_dir.iterdir() if d.is_dir() and d.name.split("__")[-1] == model_name]
    out: dict[str, dict] = {}
    for task in ("hanzi_structure", "hanzi_pinyin"):
        candidates = [
            sample_file
            for folder in matches
            for sample_file in folder.glob(f"samples_{task}_*.jsonl")
        ]
        if not candidates:
            continue
        sample_file = max(candidates, key=lambda path: path.name)
        task_predictions = {}
        with sample_file.open() as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                doc_hash = sample.get("doc_hash")
                doc_id = sample.get("doc_id")
                if not isinstance(doc_hash, str) or not doc_hash or not isinstance(doc_id, int):
                    raise ValueError(f"Missing doc_hash in {sample_file}")
                item_id = f"{doc_id}:{doc_hash}"
                if item_id in task_predictions:
                    raise ValueError(f"Duplicate Hanzi item {item_id} in {sample_file}")
                task_predictions[item_id] = _hanzi_sample_prediction(sample)
        out[task] = task_predictions
    return out


def _newest_meco_file(meco_dir: Path, model_name: str, filename: str) -> Path | None:
    """Return the newest matching MECO output file for a model in one revision dir.

    meco_py.cli writes into meco_<org__model>/ folders; match on the part after '__'.
    """
    if not meco_dir.exists():
        return None
    matches = [
        path
        for path in meco_dir.glob(f"meco_*/{filename}")
        if path.parent.name.removeprefix("meco_").split("__")[-1] == model_name
    ]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def load_meco_predictions(meco_dir: Path, model_name: str) -> dict:
    """Load raw per-word MECO surprisals written by meco_py.cli (for server scoring)."""
    pred_file = _newest_meco_file(meco_dir, model_name, "predictions_meco.json")
    if pred_file is None:
        return {}
    with pred_file.open() as f:
        data = json.load(f)
    return {key: data[key] for key in ("meco_l1", "meco_l2") if data.get(key)}


def load_meco_scores(meco_dir: Path, model_name: str) -> dict[str, float]:
    """Load locally-computed MECO delta log-likelihood scores for one revision.

    meco_py.cli writes results_meco.json in {task: {lang: score}} form, where the
    score is the delta log-likelihood — the exact metric the leaderboard recomputes
    server-side for the final checkpoint. Flatten to {meco_l1_<lang>: score, ...} so
    the values sit alongside the other zero-shot accuracies in the fast summary.
    """
    score_file = _newest_meco_file(meco_dir, model_name, "results_meco.json")
    if score_file is None:
        return {}
    with score_file.open() as f:
        data = json.load(f)
    return {
        f"{task}_{lang}": score
        for task in ("meco_l1", "meco_l2")
        for lang, score in (data.get(task) or {}).items()
    }


def load_hidden_predictions(root: Path, model_name: str, revision: str) -> dict:
    """Load server-scored Hanzi and MECO predictions for one revision."""
    return {
        **load_hanzi_predictions(root / "results" / revision, model_name),
        **load_meco_predictions(root / "meco" / "results" / revision, model_name),
    }


def main():
    parser = argparse.ArgumentParser(description="Collate multilingual evaluation results into a submission JSON.")
    parser.add_argument("--model_name", required=True,
                        help="Model folder name to match. For zeroshot results the part after '__' is matched; "
                             "for finetune results the folder name is matched directly.")
    parser.add_argument("--output",
                        help="Output JSON path (default: <model_name>_submission.json in cwd)")
    parser.add_argument("--output_predictions",
                        help="Output predictions JSON path (default: <model_name>_predictions.json in cwd)")
    parser.add_argument("--fast", action="store_true",
                        help="Also collate per-revision summaries of locally-scored intermediate "
                             "checkpoints (zeroshot accuracies plus Hanzi accuracy and MECO delta "
                             "log-likelihood) from results/<revision>/ and meco/results/<revision>/.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent  # multilingual/
    zeroshot = load_zeroshot(root / "results" / "main", args.model_name)
    finetune = load_finetune(root / "finetune" / "results", args.model_name)
    hidden_predictions = load_hidden_predictions(root, args.model_name, "main")

    warn_missing_zeroshot(
        zeroshot,
        label="main",
        extra_present=hidden_predictions,
    )
    warn_missing_finetune(finetune)

    fast_summaries: list[dict[str, float]] = []
    if args.fast:
        for rev in REVISIONS:
            # Intermediate checkpoints are scored entirely locally and submitted as
            # raw numbers — no per-checkpoint predictions are uploaded. Hanzi
            # (lm-eval accuracy) and MECO (delta log-likelihood) are folded in with
            # the same metrics the leaderboard applies to the final checkpoint.
            rev_zeroshot = load_zeroshot(
                root / "results" / rev, args.model_name, include_server_scored=True
            )
            summary = {task: subtasks[task] for task, subtasks in rev_zeroshot.items()}
            summary.update(load_meco_scores(root / "meco" / "results" / rev, args.model_name))
            warn_missing_zeroshot(rev_zeroshot, label=rev)
            fast_summaries.append(summary)

    combined = {**zeroshot, **finetune}

    if not combined and not hidden_predictions:
        print("No results or hidden-task predictions found — nothing written.")
        return

    output_path = Path(args.output) if args.output else Path(f"{args.model_name}_submission.json")
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"Wrote {len(combined)} task(s) to {output_path}")
    for task, subtasks in combined.items():
        avg = sum(subtasks.values()) / len(subtasks)
        n = len(subtasks)
        print(f"  {task}: {avg:.4f}  ({n} subtask{'s' if n > 1 else ''})")

    # --- predictions file ---
    zeroshot_raw = load_zeroshot_raw(root / "results" / "main", args.model_name)
    finetune_preds = load_finetune_predictions(root / "finetune" / "results", args.model_name)

    if zeroshot_raw or finetune_preds or hidden_predictions or fast_summaries:
        predictions = {
            "zeroshot": zeroshot_raw,
            "finetune": finetune_preds,
            "hidden": hidden_predictions,
        }
        if fast_summaries:
            predictions["fast_eval_results"] = fast_summaries
        pred_path = Path(args.output_predictions) if args.output_predictions else Path(f"{args.model_name}_predictions.json")
        with open(pred_path, "w") as f:
            json.dump(predictions, f, indent=2)
        msg = (f"Wrote predictions file to {pred_path} "
               f"({len(zeroshot_raw)} zeroshot tasks, {len(finetune_preds)} finetune task-lang pairs, "
               f"{len(hidden_predictions)} hidden tasks")
        if fast_summaries:
            msg += f", {len(fast_summaries)} fast-eval revisions"
        msg += ")"
        print(msg)


if __name__ == "__main__":
    main()
