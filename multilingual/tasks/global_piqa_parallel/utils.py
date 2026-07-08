"""Custom token-count-normalized acc_norm for Global PIQA, matching the strict harness.

lm-eval's built-in multiple_choice scorer normalizes by the *character* length of each choice
string (acc_norm) — see lm_eval/api/task.py. The strict sentence_zero_shot harness instead divides
the summed completion log-prob by the *token count* of the completion. To make the two harnesses
agree exactly, we override process_results and re-derive, per choice, the exact number of
continuation tokens lm-eval scored:

    n_tokens = len(encode(prompt + " " + choice)) - len(encode(prompt))

which equals the continuation slice lm-eval used (whole_enc[len(context_enc):]) and the strict
phrase-mask count. We count with add_special_tokens=False against the model's own tokenizer (located
via the GLOBAL_PIQA_TOKENIZER env var, set to the HF repo by scripts/global_piqa_model.sh): any
leading BOS / trailing EOS cancels in the subtraction, so the count matches lm-eval's continuation
slice whether or not the hf-bos BOS-fix is in use. The metric NAME stays `acc_norm` so the group
aggregation, collate_results.py and the markdown are unchanged; only its definition (token- vs
char-normalization) differs.
"""
import os
from functools import lru_cache

import numpy as np
from transformers import AutoTokenizer


@lru_cache(maxsize=1)
def _tokenizer():
    path = os.environ.get("GLOBAL_PIQA_TOKENIZER")
    if not path:
        raise RuntimeError(
            "GLOBAL_PIQA_TOKENIZER is not set; it must point at the model's tokenizer / HF repo "
            "(set by scripts/global_piqa_model.sh)."
        )
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def process_results(doc, results):
    lls = np.array([r[0] for r in results])  # results are in doc_to_choice (solution0, solution1, ...) order
    n = sum(1 for k in doc if k.startswith("solution"))
    choices = [doc[f"solution{i}"] for i in range(n)]

    tok = _tokenizer()
    prompt = doc["prompt"]
    ctx_len = len(tok.encode(prompt, add_special_tokens=False))  # BOS/EOS cancel in the subtraction
    tok_counts = np.array(
        [max(len(tok.encode(prompt + " " + c, add_special_tokens=False)) - ctx_len, 1) for c in choices],
        dtype=float,
    )

    pred_norm = int(np.argmax(lls / tok_counts))
    gold = doc["label"]
    return {"acc_norm": 1.0 if pred_norm == gold else 0.0}
