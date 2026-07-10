"""Scoring helpers for the Chinese Hanzi minimal-pair tasks."""

import os
from functools import lru_cache

import numpy as np
from transformers import AutoTokenizer


@lru_cache(maxsize=1)
def _tokenizer():
    model_name = os.environ.get("HANZI_TOKENIZER")
    if not model_name:
        raise RuntimeError(
            "HANZI_TOKENIZER is not set. Run Hanzi tasks through "
            "scripts/zeroshot_model.sh or set it to the model/tokenizer path."
        )
    revision = os.environ.get("HANZI_TOKENIZER_REVISION", "main")
    return AutoTokenizer.from_pretrained(
        model_name, revision=revision, trust_remote_code=True
    )


def process_results(doc, results):
    """Match the original Hanzi scorer, including its UNK-token policy."""
    tokenizer = _tokenizer()
    unk_id = tokenizer.unk_token_id
    if unk_id is not None and any(
        unk_id in tokenizer.encode(sentence, add_special_tokens=False)
        for sentence in (doc["sentence_good"], doc["sentence_bad"])
    ):
        return {"acc": 0.0}

    log_likelihoods = np.array([result[0] for result in results])
    # gold is index 0; an exact tie (np.argmax would break it toward 0) counts
    # as wrong. (UNK pairs are already returned as 0.0 above.)
    top = np.flatnonzero(log_likelihoods == log_likelihoods.max())
    return {"acc": 1.0 if top.size == 1 and top[0] == 0 else 0.0}
