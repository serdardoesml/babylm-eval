"""No-R front-end for MECO: reproduce meco_l2.R / meco_l1.R data prep in pandas.

Builds, per reader language, the exact data frame the R fitters feed to lmer --
from the .rda reading times + a surprisal TSV, with wordfreq zipf frequencies.
The L2 frame was validated against R's fit frame as an exact multiset of rows;
per-language early filtering is used for speed and is equivalent because every
L2 reader sees the same English stimulus (itemid, wordnum, word are shared).
"""
import numpy as np
import pandas as pd
import pyreadr
import wordfreq

# L2 reader-group label -> language code in the .rda (meco_l2.R lang_map).
L2_LANG_MAP = {"nld": "du", "deu": "ge", "zho": "ch_s"}
# L1 language code -> wordfreq language (meco_l1.R wf_lang).
L1_WF_LANG = {"en": "en", "du": "nl", "ch_s": "zh"}

RE_TERMS_FULL = ["Surprisal", "Surprisal1p", "Surprisal2p", "wordnum",
                 "wlen", "wlen1p", "wlen2p", "freq", "freq1p", "freq2p"]
RE_TERMS_BASE = ["wordnum", "wlen", "wlen1p", "wlen2p", "freq", "freq1p", "freq2p"]


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_l2_all_data(data_dir):
    w1 = pyreadr.read_r(f"{data_dir}/joint_data_l2_trimmed_version2.2.rda")["joint.data"]
    w2 = pyreadr.read_r(f"{data_dir}/joint_data_trimmed_L2_wave2_version2.2.rda")["joint.data"]
    w2 = w2.drop(columns=["supplementary_id"])         # meco_l2.R: select(-supplementary_id)
    return pd.concat([w1, w2[w1.columns]], ignore_index=True)


def load_l1_all_data(data_dir):
    w1 = pyreadr.read_r(f"{data_dir}/wave1_joint_l1_data_trimmed_version2.1.rda")["joint.data"]
    w2 = pyreadr.read_r(f"{data_dir}/wave2_joint_data_trimmed_wave2_version2.1.rda")["joint.data"]
    w2 = w2.drop(columns=["trialnum"])                 # meco_l1.R: select(-trialnum)
    return pd.concat([w1, w2[w1.columns]], ignore_index=True)


def read_surprisal(surprisal_tsv, extra_id_cols=()):
    lm = pd.read_csv(surprisal_tsv, sep="\t")
    drop = [c for c in ("Unnamed: 0", "...1", "FullText", "FullTextMarked")
            if c in lm.columns]
    lm = lm.drop(columns=drop)
    lm["itemid"] = lm["itemid"].astype(str)
    # readr::read_tsv (which the R fitters use) defaults to trim_ws=TRUE, so it
    # strips leading/trailing whitespace from character fields. The rda `word`
    # is loaded untrimmed; matching readr here is what leaves the few
    # whitespace-padded IAs (e.g. a Chinese word with a trailing space)
    # unmatched, exactly as the reference pipeline does.
    lm["word"] = lm["word"].astype(str).str.strip()
    return lm


# --------------------------------------------------------------------------- #
# spillover (get_prev) -- 1- and 2-word lag on (itemid, wordnum)
# --------------------------------------------------------------------------- #
def _get_prev(cur):
    def back(shift, suf):
        b = (cur[["itemid", "wordnum", "Surprisal", "wlen", "freq"]]
             .drop_duplicates()
             .rename(columns={"Surprisal": f"Surprisal{suf}",
                              "wlen": f"wlen{suf}", "freq": f"freq{suf}"}))
        b["wordnum"] = b["wordnum"] - shift
        return b[b["wordnum"] >= 1].drop_duplicates()
    out = cur.merge(back(1, "1p"), on=["itemid", "wordnum"], how="inner")
    out = out.merge(back(2, "2p"), on=["itemid", "wordnum"], how="inner")
    return out.drop_duplicates()


def _trim_and_join(ad):
    """group_by(itemid, sentnum) drop first/last word; inner-join spillover.

    dropna=False mirrors dplyr's group_by, which keeps NA groups: some waves
    carry NA sentnum, and R still trims the min/max wordnum within the
    (itemid, NA) group. pandas' default dropna=True would skip those rows and
    leave sentence-edge words in (observed as a large zho divergence).
    """
    g = ad.groupby(["itemid", "sentnum"], dropna=False)["wordnum"]
    sub = ad[~ad["wordnum"].eq(g.transform("min"))
             & ~ad["wordnum"].eq(g.transform("max"))]
    prev = _get_prev(ad)
    common = [c for c in sub.columns if c in prev.columns]
    return sub.merge(prev, on=common, how="inner")


# --------------------------------------------------------------------------- #
# L2 frame (English stimulus; frequency in English)
# --------------------------------------------------------------------------- #
def l2_word_table(all_data):
    words = pd.DataFrame({"word": all_data["word"].drop_duplicates().values})
    words["wlen"] = words["word"].str.len()
    words["freq"] = [wordfreq.zipf_frequency(w, "en") for w in words["word"]]
    return words


def _prep_l2(all_data, lm, model_col, words):
    ad = all_data.copy()
    ad["itemid"] = ad["itemid"].astype(str)
    ad = ad.merge(words, on="word", how="left")
    ad = ad.merge(lm[["itemid", "wordnum", "word", model_col]],
                  on=["itemid", "wordnum", "word"], how="left")
    return ad.rename(columns={model_col: "Surprisal"})


def build_l2_frames(all_data, lm, model_col, lang_codes, words=None):
    """All requested L2 reader-language frames at once.

    meco_l2.R fits on the full multi-language data (every reader saw the same
    English stimulus): the sentence-trim and get_prev spillover are computed
    over all languages, then rows are filtered to each reader group. We compute
    that shared frame once and slice it per language -- equivalent to R and much
    faster than repeating the joins per language.
    """
    if words is None:
        words = l2_word_table(all_data)
    ad = _prep_l2(all_data, lm, model_col, words)
    fit_all = _trim_and_join(ad)
    return {code: fit_all[fit_all["lang"] == code].copy() for code in lang_codes}


def build_l2_frame(all_data, lm, model_col, lang_code, words=None):
    return build_l2_frames(all_data, lm, model_col, [lang_code], words)[lang_code]


# --------------------------------------------------------------------------- #
# L1 frame (texts differ by language; frequency in the text's language)
# --------------------------------------------------------------------------- #
def l1_word_table(all_data, lang_codes):
    ad = all_data[all_data["lang"].isin(lang_codes)]
    words = ad[["lang", "word"]].drop_duplicates().copy()
    words["wlen"] = words["word"].str.len()
    words["freq"] = [wordfreq.zipf_frequency(w, L1_WF_LANG[l])
                     for w, l in zip(words["word"], words["lang"])]
    return words


def build_l1_frame(all_data, lm, model_col, lang_code, words=None):
    if words is None:
        words = l1_word_table(all_data, [lang_code])
    ad = all_data[all_data["lang"] == lang_code].copy()
    ad["itemid"] = ad["itemid"].astype(str)
    ad = ad.merge(words[words["lang"] == lang_code], on=["lang", "word"], how="left")
    join_cols = ["lang", "itemid", "wordnum", "word"]
    ad = ad.merge(lm[join_cols + [model_col]], on=join_cols, how="left")
    ad = ad.rename(columns={model_col: "Surprisal"})
    return _trim_and_join(ad)
