"""Regression test locking all 36 MECO L1/L2 cells to the R/lme4 reference.

Runs the pure-Python stats pipeline (frame + fit) on frozen surprisal inputs and
asserts each ``normalized_loglik`` matches the committed R gold. This locks the
R -> Python port: surprisal computation (minicons) is out of scope here and is
exercised separately.

Fixtures (self-contained, no model downloads, no R):
  * tests/fixtures/surprisal/{l1,l2}_surprisal_<stem>.tsv.gz  -- frozen inputs
  * tests/gold_normalized_loglik.csv                          -- frozen R gold
  * ../data/{meco_l1,meco_l2}/*.rda                           -- reading times

Run:  pytest multilingual/meco/tests/test_regression.py -v
"""
import pathlib
import sys

import pandas as pd
import pytest

HERE = pathlib.Path(__file__).resolve().parent
MECO = HERE.parent
sys.path.insert(0, str(MECO.parent.parent))          # repo root, so `meco` imports

from meco.meco_py import (  # noqa: E402
    load_l1_all_data, load_l2_all_data, read_surprisal,
    build_l1_frame, build_l2_frames, l1_word_table, l2_word_table,
    normalized_loglik, L2_LANG_MAP,
)

# Max |py - R| across all 36 cells is ~5.1e-3 (optimizer-stop noise); 1e-2 gives
# headroom while staying orders of magnitude below inter-model score gaps.
TOL = 1e-2

GOLD = pd.read_csv(HERE / "gold_normalized_loglik.csv")
FIX = HERE / "fixtures" / "surprisal"
DATA = MECO / "data"


@pytest.fixture(scope="session")
def l1_data():
    ad = load_l1_all_data(str(DATA / "meco_l1"))
    return ad, l1_word_table(ad, ["en", "du", "ch_s"])


@pytest.fixture(scope="session")
def l2_data():
    ad = load_l2_all_data(str(DATA / "meco_l2"))
    return ad, l2_word_table(ad)


def _norm(row, l1_data, l2_data):
    stem, lang, ev = row["stem"], row["lang"], row["eval"]
    col = stem.replace("-", "__")
    lm = read_surprisal(FIX / f"{ev.lower()}_surprisal_{stem}.tsv.gz")
    if ev == "L1":
        lm["lang"] = lm["lang"].astype(str)
        ad, words = l1_data
        fr = build_l1_frame(ad, lm, col, lang, words)
    else:
        ad, words = l2_data
        code = L2_LANG_MAP[lang]
        fr = build_l2_frames(ad, lm, col, [code], words)[code]
    return normalized_loglik(fr)[2]


@pytest.mark.parametrize(
    "row",
    [r for _, r in GOLD.iterrows()],
    ids=[f"{r['eval']}-{r['stem']}-{r['lang']}" for _, r in GOLD.iterrows()],
)
def test_cell_matches_reference(row, l1_data, l2_data):
    got = _norm(row, l1_data, l2_data)
    exp = row["normalized_loglik"]
    assert abs(got - exp) <= TOL, (
        f"{row['eval']} {row['stem']} {row['lang']}: "
        f"python={got:.6f} R={exp:.6f} |delta|={abs(got - exp):.2e} > {TOL}")
