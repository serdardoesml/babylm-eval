"""In-repo lm-eval HF model that strips auto special tokens and (optionally) prepends BOS.

Registered as `hf-bos`. Use it when a model's tokenizer over-adds special tokens on every
encode() -- e.g. the BabyLM GPT2 baselines whose post-processor wraps every string in
`<s> ... </s>`. The trailing `</s>` corrupts lm-eval's context/continuation split (it drops the
real first continuation token and scores a spurious `</s>`); the leading `<s>` is *wanted* -- the
strict sentence_zero_shot harness scores completions conditioned on a single leading `<s>`.

This subclass tokenizes with `add_special_tokens=False` (dropping the spurious `</s>`) and prepends
exactly one BOS when `add_bos_token=True`. Because both the `whole` (prompt+continuation) and the
`context` (prompt) encodings get the same leading BOS, lm-eval's `whole[len(ctx):]` continuation
split cancels it -- leaving a clean, BOS-conditioned continuation identical to the strict harness,
WITHOUT a separate BOS-only tokenizer artifact on disk.

Scope: the loglikelihood / multiple_choice path (all BabyLM zeroshot + Global PIQA tasks). Generation
tasks are not in scope. Participants whose tokenizer is already correct (BOS-only, or assumes no BOS)
do NOT need this model -- they can use the stock `--model hf` (optionally `--bos_fix 0` in the runner
scripts).
"""
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("hf-bos")
class BosOnlyHFLM(HFLM):
    def tok_encode(self, string, add_special_tokens=None, left_truncate_len=None, **kwargs):
        enc = self.tokenizer.encode(string, add_special_tokens=False)  # drop spurious </s>
        if self.add_bos_token and self.prefix_token_id is not None:
            enc = [self.prefix_token_id] + enc                         # prepend single <s>
        if left_truncate_len:
            enc = enc[-left_truncate_len:]
        return enc
