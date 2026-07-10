"""Shared tie-aware scoring for the multilingual zeroshot log-prob tasks.

This is the single source of truth. It is symlinked into each task directory as
``tie_handling.py`` because lm-eval's ``!function`` loads the module from the
yaml's *own* directory -- edit this file and every task picks it up.

lm-eval's built-in multiple_choice ``acc`` uses ``np.argmax``, which breaks ties
toward the first choice. On tie-heavy models (e.g. UNK collapse) that silently
favours whichever answer sits at index 0. Here a k-way tie for the top score
gets ``1/k`` credit (0.5 for the usual 2-way tie): the fraction of an even split
over the tied choices that lands on gold -- the deterministic equivalent of a
coin flip on ties, with no RNG.

    process_results          -- minimal-pair tasks (gold is index 0); acc only.
    process_results_labeled  -- labelled multiple choice (gold = doc["label"]);
                                acc and acc_norm (length-normalised).
"""

import numpy as np


def _fractional(scores, gold):
    top = np.flatnonzero(scores == scores.max())
    # k-way tie including gold => 1/k credit; gold not tied for top => wrong.
    return (1.0 / top.size) if gold in top else 0.0


def process_results(doc, results):
    lls = np.array([r[0] for r in results], dtype=float)
    return {"acc": _fractional(lls, 0)}


def _gold_index(doc, choices):
    gold = doc["label"]
    if isinstance(gold, str):
        if gold in choices:
            return choices.index(gold)
        return int(gold) if gold.lstrip("-").isdigit() else -100
    return int(gold)


def process_results_labeled(doc, results):
    lls = np.array([r[0] for r in results], dtype=float)
    choices = doc["choices"]
    gold = _gold_index(doc, choices)
    completion_len = np.array([float(len(c)) for c in choices])
    return {
        "acc": _fractional(lls, gold),
        "acc_norm": _fractional(lls / completion_len, gold),
    }
