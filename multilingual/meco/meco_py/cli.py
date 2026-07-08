"""MECO delta log-likelihood evaluation for the BabyLM multilingual track.

Same ergonomics as the other multilingual tasks (``--model_name`` / ``--langs``
/ ``--revision``), with the track language codes ``eng`` / ``nld`` / ``zho``:

    python -m meco.meco_py.cli --model_name ORG/MODEL --langs "eng nld zho"

For each requested language it computes two MECO measures where applicable:

  * ``meco_l1`` -- native reading (eng->en, nld->du, zho->ch_s), one score per
    language in the model's training mixture.
  * ``meco_l2`` -- reading English as a second language, scored per non-English
    reader group (nld, zho); run when English is in the mixture.

Surprisal is computed with minicons; the delta log-likelihood mixed models are
fit in pure Python (no R / lme4). Outputs, matching the track conventions:

  * ``<out_dir>/<revision>/meco_<org__model>/results_<...>.json`` -- per-model
    scores in the collation-friendly ``{task: {lang: score}}`` submission schema.
  * a combined ``meco_delta_loglik_<model>.csv`` with full logLik detail.

The submission score is the ``normalized_loglik`` delta log-likelihood
(logLik_full - logLik_base) -- the *same* quantity the leaderboard recomputes
server-side for the final checkpoint (``_score_meco``). Emitting it here keeps the
locally-scored intermediate checkpoints on an identical metric scale to the final
point. The McFadden ``proportion_deviance_explained`` and full logLik detail are
kept in the CSV. Incomplete evaluation is allowed: languages the model does not
cover are skipped.
"""
import argparse
import json
import math
import pathlib
import sys

import pandas as pd

from . import (
    load_l1_all_data, load_l2_all_data,
    build_l1_frame, build_l2_frames, l1_word_table, l2_word_table,
    normalized_loglik, L2_LANG_MAP,
)
from .frame import read_surprisal
from .surprisal import compute_surprisal, surprisal_column_name

HERE = pathlib.Path(__file__).resolve().parents[1]          # multilingual/meco

# Track language code (eng/nld/zho) -> MECO native-reading (L1) language code.
TRACK_TO_L1 = {"eng": "en", "nld": "du", "zho": "ch_s"}
# Track language -> MECO L2 reader group (English-as-L2). English is the text
# being read, so it has no L2 reader group of its own.
TRACK_TO_L2 = {"nld": "nld", "zho": "zho"}
# Which track languages a model was trained on, inferred from its name.
NAME_HINTS = {
    "eng": ("en_", "_en", "-en", "eng", "strict", "interaction"),
    "nld": ("nld", "_nl", "-nl", "nl_"),
    "zho": ("zho", "_zh", "-zh", "zh_"),
}


def mixture_langs(name):
    n = name.lower()
    return [lg for lg, hints in NAME_HINTS.items() if any(h in n for h in hints)]


def parse_langs(s):
    return [x for x in (s or "").replace(",", " ").split() if x]


def _score(fr):
    ll_full, ll_base, norm = normalized_loglik(fr)
    # ``norm`` (logLik_full - logLik_base) is the leaderboard's server-side MECO
    # metric (_score_meco). It is submitted directly so intermediate checkpoints
    # share the final point's metric scale. McFadden's proportion of deviance
    # explained is retained in the detail CSV only.
    pde = 1.0 - ll_full / ll_base
    return dict(logLik=ll_full, baseline_loglik=ll_base, normalized_loglik=norm,
                proportion_deviance_explained=pde)


def _id_part(value):
    """Serialize pandas numeric IDs without unstable trailing '.0'."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(numeric)) if numeric.is_integer() else str(value)


def _prediction_map(frame, model_col):
    """Return stable item-position IDs mapped to raw model surprisal.

    A few MECO items share an ``(itemid, wordnum)`` position across hyphenation
    variants of the same word (e.g. "Grief-"/"Grief"). The reading data and the
    server manifest both key on ``itemid:wordnum`` alone, which cannot represent
    that distinction, so such rows collapse to a single prediction ID. We keep
    the first occurrence and warn, rather than erroring, since the extra rows
    cannot be uploaded under this ID scheme anyway.
    """
    predictions = {}
    for itemid, wordnum, raw_value in frame[
        ["itemid", "wordnum", model_col]
    ].itertuples(index=False, name=None):
        key = f"{_id_part(itemid)}:{_id_part(wordnum)}"
        if key in predictions:
            print(f"Warning: duplicate MECO prediction ID {key}; keeping first occurrence.",
                  file=sys.stderr)
            continue
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite MECO surprisal for {key}")
        predictions[key] = value
    return predictions


def evaluate(model_name, track_langs, do_l2, device, data_dir,
             l1_stims, l2_stims, l1_surprisal=None, l2_surprisal=None,
             backend="auto", revision=None, trust_remote_code=True, batch_size=32):
    """Return local diagnostics plus raw predictions for server-side scoring."""
    col = surprisal_column_name(model_name)
    submission = {"meco_l1": {}, "meco_l2": {}}
    predictions = {"meco_l1": {}, "meco_l2": {}}
    detail = []

    l1_track = [lg for lg in track_langs if lg in TRACK_TO_L1]
    if l1_track:
        codes = [TRACK_TO_L1[lg] for lg in l1_track]
        if l1_surprisal is not None:
            lm = read_surprisal(l1_surprisal)
        else:
            stims = pd.read_csv(l1_stims, sep="\t")
            stims = stims[stims["lang"].isin(codes)].reset_index(drop=True)
            sup, _ = compute_surprisal(model_name, stims, device, backend, revision,
                                       trust_remote_code, batch_size)
            sup.to_csv(pathlib.Path(l1_stims).with_suffix(".surprisal.tsv"), sep="\t", index=False)
            lm = _frame_from_surprisal(sup)
        all_data = load_l1_all_data(str(pathlib.Path(data_dir) / "meco_l1"))
        words = l1_word_table(all_data, codes)
        for lg, code in zip(l1_track, codes):
            predictions["meco_l1"][lg] = _prediction_map(
                lm[lm["lang"] == code], col
            )
            fr = build_l1_frame(all_data, lm, col, code, words)
            s = _score(fr)
            submission["meco_l1"][lg] = s["normalized_loglik"]
            detail.append(dict(eval="L1", model_id=model_name, lang=lg, meco_lang=code, **s))

    l2_track = [lg for lg in track_langs if lg in TRACK_TO_L2] if do_l2 else []
    if l2_track:
        if l2_surprisal is not None:
            lm = read_surprisal(l2_surprisal)
        else:
            stims = pd.read_csv(l2_stims, sep="\t")
            sup, _ = compute_surprisal(model_name, stims, device, backend, revision,
                                       trust_remote_code, batch_size)
            sup.to_csv(pathlib.Path(l2_stims).with_suffix(".surprisal.tsv"), sep="\t", index=False)
            lm = _frame_from_surprisal(sup)
        all_data = load_l2_all_data(str(pathlib.Path(data_dir) / "meco_l2"))
        words = l2_word_table(all_data)
        codes = [TRACK_TO_L2[lg] for lg in l2_track]
        frames = build_l2_frames(all_data, lm, col, [L2_LANG_MAP[c] for c in codes], words)
        l2_predictions = _prediction_map(lm, col)
        for lg, c in zip(l2_track, codes):
            predictions["meco_l2"][lg] = l2_predictions
            fr = frames[L2_LANG_MAP[c]]
            s = _score(fr)
            submission["meco_l2"][lg] = s["normalized_loglik"]
            detail.append(dict(eval="L2", model_id=model_name, lang=lg, meco_lang=c, **s))

    submission = {task: scores for task, scores in submission.items() if scores}
    predictions = {task: values for task, values in predictions.items() if values}
    return submission, detail, predictions


def _frame_from_surprisal(sup_df):
    df = sup_df.copy()
    drop = [c for c in ("Unnamed: 0", "...1", "FullText", "FullTextMarked") if c in df.columns]
    df = df.drop(columns=drop)
    df["itemid"] = df["itemid"].astype(str)
    if "lang" in df.columns:
        df["lang"] = df["lang"].astype(str)
    df["word"] = df["word"].astype(str).str.strip()   # mirror readr trim_ws
    return df


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", required=True, help="HuggingFace model id.")
    ap.add_argument("--langs", default="eng nld zho", help='Track languages (default all).')
    ap.add_argument("--revision", default="main", help="Checkpoint/revision (default main).")
    ap.add_argument("--l2", default="auto", choices=["auto", "on", "off"],
                    help="Run MECO L2 (English-as-L2). auto = on iff English in the mixture.")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "causal", "mlm", "seq2seq", "mamba"],
                    help="minicons scorer. auto detects from the model config.")
    ap.add_argument("--no_trust_remote_code", action="store_true",
                    help="Do not pass trust_remote_code=True when loading.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=16,
                    help="Causal tail-window batch size (1 = original per-window passes).")
    ap.add_argument("--data_dir", default=str(HERE / "data"))
    ap.add_argument("--l1_stims", default=str(HERE / "data" / "meco_l1_stims.tsv"))
    ap.add_argument("--l2_stims", default=str(HERE / "data" / "meco_l2_stims.tsv"))
    ap.add_argument("--l1_surprisal", default=None,
                    help="Precomputed L1 surprisal TSV (skip surprisal computation).")
    ap.add_argument("--l2_surprisal", default=None,
                    help="Precomputed L2 surprisal TSV (skip surprisal computation).")
    ap.add_argument("--out_dir", default=str(HERE / "results"), type=pathlib.Path)
    ap.add_argument("--output", default=None, help="Combined detail CSV path.")
    args = ap.parse_args()

    stem = args.model_name.split("/")[-1]
    org_model = args.model_name.replace("/", "__")
    requested = parse_langs(args.langs)
    mix = mixture_langs(args.model_name) or requested         # fall back to requested
    track_langs = [lg for lg in requested if lg in mix] or requested
    do_l2 = {"on": True, "off": False, "auto": "eng" in mix}[args.l2]
    print(f"=== MECO {stem} ({args.revision}) | langs: {track_langs} | L2: {do_l2} ===",
          flush=True)

    submission, detail, predictions = evaluate(
        args.model_name, track_langs, do_l2, args.device, args.data_dir,
        args.l1_stims, args.l2_stims, args.l1_surprisal, args.l2_surprisal,
        backend=args.backend, revision=args.revision,
        trust_remote_code=not args.no_trust_remote_code, batch_size=args.batch_size)

    if not detail:
        print("No MECO evaluations were run.", file=sys.stderr)
        return

    # Submission-schema JSON under results/<revision>/meco_<org__model>/ so it
    # sits alongside the other multilingual results and is easy to collate.
    res_dir = args.out_dir / args.revision / f"meco_{org_model}"
    res_dir.mkdir(parents=True, exist_ok=True)
    sub_path = res_dir / "results_meco.json"
    with open(sub_path, "w") as f:
        json.dump(submission, f, indent=2)

    # This is the leaderboard input. Final MECO scores are recomputed
    # server-side from these per-word surprisals.
    pred_path = res_dir / "predictions_meco.json"
    with pred_path.open("w") as f:
        json.dump(predictions, f)

    detail_df = pd.DataFrame(detail)
    out = args.output or str(res_dir / f"meco_delta_loglik_{stem}.csv")
    detail_df.to_csv(out, index=False)

    print(json.dumps(submission, indent=2), flush=True)
    print(f"\nwrote {sub_path}\nwrote {pred_path}\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
