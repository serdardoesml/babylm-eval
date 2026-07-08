"""Pure-Python MECO delta log-likelihood evaluation (no R / lme4 required).

Faithful port of meco_l2.R / meco_l1.R, validated against lme4 4.x to ~1e-6 in
normalized_loglik on the BabyLM-2026 baselines.
"""
from .fit import loglik, normalized_loglik
from .frame import (
    load_l2_all_data, load_l1_all_data, read_surprisal,
    build_l2_frame, build_l2_frames, build_l1_frame,
    l2_word_table, l1_word_table,
    L2_LANG_MAP, L1_WF_LANG,
)

__all__ = [
    "loglik", "normalized_loglik",
    "load_l2_all_data", "load_l1_all_data", "read_surprisal",
    "build_l2_frame", "build_l2_frames", "build_l1_frame",
    "l2_word_table", "l1_word_table",
    "L2_LANG_MAP", "L1_WF_LANG",
]
