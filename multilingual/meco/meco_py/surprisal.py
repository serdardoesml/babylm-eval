"""Per-word surprisal for MECO stimuli, computed with minicons.

Faithful to the gold-standard MECO surprisal computation: each stimulus row
carries a ``FullTextMarked`` string with the target word wrapped in ``*...*``;
surprisal is the summed token surprisal of the target word conditioned on its
prefix (``reduction = -x.sum(0)``).

Architecture-general. The ``backend`` selects the minicons scorer; ``auto``
detects it from the model config so masked, causal, mixture-of-experts,
encoder-decoder and state-space (Mamba) models all work:

  * ``causal``   -> IncrementalLMScorer  (GPT-2, Llama, Mistral, and MoE decoders
                                          such as Mixtral / OLMoE / Qwen-MoE)
  * ``mlm``      -> MaskedLMScorer        (BERT-style, GPT-BERT masked head)
  * ``seq2seq``  -> Seq2SeqScorer         (T5 / BART / mT5)
  * ``mamba``    -> MambaScorer           (Mamba / state-space models)

MoE is not a separate scorer: a mixture-of-experts model is loaded through the
same causal (or masked) path as its dense counterpart, so it needs no special
handling here beyond ``trust_remote_code`` for custom code.
"""
import bisect
from collections import defaultdict

import pandas as pd
import torch
from transformers import AutoConfig
from minicons import scorer
from tqdm import tqdm

_SCORERS = {
    "causal": scorer.IncrementalLMScorer,
    "mlm": scorer.MaskedLMScorer,
    "seq2seq": scorer.Seq2SeqScorer,
    "mamba": scorer.MambaScorer,
}


def detect_backend(model_name, revision=None, trust_remote_code=True):
    """Infer the minicons backend from a model's config."""
    cfg = AutoConfig.from_pretrained(
        model_name, revision=revision, trust_remote_code=trust_remote_code)
    archs = [a.lower() for a in (getattr(cfg, "architectures", None) or [])]
    mtype = (getattr(cfg, "model_type", "") or "").lower()

    def any_arch(*subs):
        return any(any(s in a for s in subs) for a in archs)

    if getattr(cfg, "is_encoder_decoder", False) or any_arch("forconditionalgeneration",
                                                             "seq2seq"):
        return "seq2seq"
    if any_arch("formaskedlm", "formacaronlm") or mtype in {"bert", "roberta",
                                                            "electra", "deberta",
                                                            "deberta-v2"}:
        return "mlm"
    if "mamba" in mtype or any_arch("mamba"):
        return "mamba"
    # Everything else (incl. MoE decoders and custom GPT-BERT causal heads).
    return "causal"


def model_max_len(ilm_model):
    cfg = ilm_model.model.config
    for attr in ("n_positions", "n_ctx", "max_position_embeddings"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and 0 < v < 1_000_000:
            return v
    return 1024


def make_get_surprisal(ilm_model, backend, max_len, margin=8):
    tok = ilm_model.tokenizer

    # The causal path is scored directly (see _make_causal_get_surprisal): the
    # BabyLM tokenizers auto-wrap every string in ``<s> ... </s>``, which breaks
    # minicons' preamble-length slicing (it drops the first target token and
    # scores a spurious ``</s>``). We construct the input by hand instead.
    if backend == "causal":
        return _make_causal_get_surprisal(ilm_model, tok, max_len, margin)

    def get_surprisal(marked_text):
        parts = marked_text.split("*")
        prefix, target = parts[0].strip(), parts[1].strip()
        # Left-truncate an over-long prefix to the most recent tokens so the
        # (prefix + target) fits the model context. Only triggers on passages
        # longer than the window; harmless for models with no hard limit.
        tgt_n = len(tok(target, add_special_tokens=False)["input_ids"])
        budget = max_len - tgt_n - margin
        pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        if budget > 0 and len(pre_ids) > budget:
            prefix = tok.decode(pre_ids[-budget:])
        # First word of a passage has an empty prefix. Encoder-decoder models
        # cannot take an empty source, so give them a minimal one.
        if backend == "seq2seq" and not prefix:
            prefix = tok.pad_token or tok.eos_token or tok.bos_token or " "
        return ilm_model.conditional_score(
            prefix, target, reduction=lambda x: -x.sum(0).item())[0]

    return get_surprisal


def _make_causal_get_surprisal(ilm_model, tok, max_len, margin):
    """Direct per-word surprisal for causal LMs, mirroring the strict
    ``sentence_zero_shot`` causal construction: tokenize with
    ``add_special_tokens=False`` (no spurious ``</s>``), prepend exactly one BOS
    when the tokenizer defines one, and locate the target tokens by character
    offset rather than by length subtraction.
    """
    model = ilm_model.model
    device = next(model.parameters()).device
    # Only an explicitly-set BOS counts -- do not alias any other special token
    # as a BOS. Without one, the passage's first token is scored with genuinely
    # empty context (and is therefore unscorable -- see below).
    bos_index = [tok.bos_token_id] if tok.bos_token_id is not None else []

    def get_surprisal(marked_text):
        parts = marked_text.split("*")
        # Keep exact characters (no strip) so offsets stay aligned. FullTextMarked
        # is the passage up to and including the starred target word.
        prefix, target = parts[0], parts[1]
        full_text = prefix + target
        cut = len(prefix)  # char index where the target begins

        enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        # Target tokens are those extending past the prefix's character span.
        target_positions = [i for i, (s, e) in enumerate(offsets) if e > cut]
        if not target_positions:
            return 0.0
        first_tgt = target_positions[0]
        n_target = len(ids) - first_tgt

        # Left-truncate prefix tokens so bos + prefix + target fits the window.
        max_prefix = max_len - len(bos_index) - n_target - margin
        prefix_ids = ids[:first_tgt]
        if max_prefix <= 0:
            prefix_ids = []
        elif len(prefix_ids) > max_prefix:
            prefix_ids = prefix_ids[len(prefix_ids) - max_prefix:]

        input_ids = bos_index + prefix_ids + ids[first_tgt:]
        start = len(input_ids) - n_target  # first scored (target) position

        ids_t = torch.tensor([input_ids], device=device)
        with torch.no_grad():
            logits = model(ids_t).logits[0]
        logp = torch.log_softmax(logits.float(), dim=-1)

        total = 0.0
        for p in range(start, len(input_ids)):
            # logits at position p-1 predict the token at position p.
            if p == 0:
                # No BOS and no context: the very first token is unscorable.
                continue
            total += logp[p - 1, input_ids[p]].item()
        return -total

    return get_surprisal


def surprisal_column_name(model_name):
    """Gold-standard column naming: org/model-name -> model__name."""
    return model_name.split("/")[-1].replace("-", "__")


def load_scorer(model_name, backend="auto", device="cpu", revision=None,
                trust_remote_code=True):
    if backend == "auto":
        backend = detect_backend(model_name, revision, trust_remote_code)
    kwargs = {"tokenizer": model_name, "trust_remote_code": trust_remote_code}
    if revision is not None:
        kwargs["revision"] = revision
    ilm = _SCORERS[backend](model_name, device, **kwargs)
    return ilm, backend


def _compute_causal_surprisal_by_passage(ilm_model, df, max_len, margin=8, batch_size=32):
    """Per-word surprisal for causal LMs, scoring a whole passage per forward pass.

    Equivalent in result to calling the per-word scorer
    (:func:`_make_causal_get_surprisal`) on every row, but with ~N times fewer
    forward passes. For a causal LM the logits at position ``i`` depend only on
    the tokens ``<= i`` and their absolute positions, so a single pass over the
    whole passage yields every word's in-context surprisal at once.

    A passage's rows share one cumulative text: ``FullText`` for word ``k`` is
    the passage up to and including word ``k`` (so the longest ``FullText`` in a
    ``(lang, itemid)`` group is the full passage, and every word is a character
    slice of it). We tokenize that passage **once** and drive all passes off the
    same token sequence, so head and tail share a single tokenization (matters
    for languages such as Chinese, where re-tokenizing a truncated
    ``prefix+target`` string could shift BPE merges at the word boundary).

    Passages longer than the context window are handled without losing context:
    a **head** pass scores every word that fits in the first ``budget`` tokens
    with its true full prefix, then each **tail** word past the window gets one
    right-anchored pass keeping the most-recent ``budget`` tokens -- the same
    truncation the per-word path applies.

    Tail windows are all right-anchored to exactly ``budget`` tokens (the sole
    exception being a single target word longer than ``budget``), so they are the
    same length and stack into one rectangular tensor with **no padding** -- we
    batch up to ``batch_size`` of them per forward pass. Batching stays within a
    passage (each passage owns its own tokenization). Passes per passage =
    ``1 + ceil(#tail words / batch_size)`` (just ``1`` when the passage fits).
    ``batch_size=1`` reproduces the original one-window-per-pass behaviour.
    """
    tok = ilm_model.tokenizer
    model = ilm_model.model
    device = next(model.parameters()).device
    # Only an explicitly-set BOS counts -- never alias another special token.
    bos_index = [tok.bos_token_id] if tok.bos_token_id is not None else []
    budget = max_len - len(bos_index) - margin

    # Per-row fallback for pathological passages whose cumulative FullText is not
    # a clean character-prefix of the target words (should not happen in MECO).
    fallback = _make_causal_get_surprisal(ilm_model, tok, max_len, margin)

    result = pd.Series(0.0, index=df.index, dtype=float)
    keys = [c for c in ("lang", "itemid") if c in df.columns]
    grouped = df.groupby(keys, sort=False) if keys else [(None, df)]

    for _, g in tqdm(grouped, desc=f"[causal, {len(grouped)} passages]"):
        g = g.sort_values("wordnum")
        passage_text = max(g["FullText"], key=len)  # = the full passage

        # Recover each word's [start, end) character span from its own marker.
        word_rows = []  # (row_index, start, end)
        ok = True
        for row_idx, marked in zip(g.index, g["FullTextMarked"]):
            parts = marked.split("*")
            if len(parts) < 2:
                ok = False
                break
            prefix, target = parts[0], parts[1]
            start = len(prefix)
            end = start + len(target)
            if passage_text[start:end] != target:
                ok = False
                break
            word_rows.append((row_idx, start, end))
        if not ok:
            for row_idx, marked in zip(g.index, g["FullTextMarked"]):
                result[row_idx] = fallback(marked)
            continue

        enc = tok(passage_text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        T = len(ids)
        token_ends = [e for (_, e) in offsets]

        def score_batch(specs):
            """One forward pass over a batch of equal-length windows; sum target
            logprobs. ``specs`` is a list of ``(a, b, words)`` with all ``b - a``
            equal (so ``bos + ids[a:b]`` stack into a rectangular tensor -- no
            padding). ``words`` is ``[(row_idx, positions), ...]``."""
            input_ids = [bos_index + ids[a:b] for (a, b, _) in specs]
            ids_t = torch.tensor(input_ids, device=device)
            with torch.no_grad():
                logits = model(ids_t).logits  # [B, T, V]
            logp = torch.log_softmax(logits.float(), dim=-1)
            for r, (a, b, words) in enumerate(specs):
                for row_idx, positions in words:
                    total = 0.0
                    for p in positions:
                        loc = len(bos_index) + p - a  # position within this window
                        if loc == 0:
                            # First token, no BOS: unscorable (no left context).
                            continue
                        total += logp[r, loc - 1, ids[p]].item()
                    result[row_idx] = -total

        # Assign each word its target token positions by the token's END char,
        # matching the per-word rule "a token is target iff token.end > cut": word
        # k owns the tokens whose end char lands in ``(start_k, end_k]``. This
        # drops lone inter-word whitespace tokens (whose end sits in the gap) and
        # assigns every real token to exactly one word. ``token_ends`` is
        # non-decreasing, so a bisect isolates each word's tokens independently
        # (robust to repeated-span rows).
        head_words, tail_words = [], []
        for row_idx, start, end in word_rows:
            lo = bisect.bisect_right(token_ends, start)  # first token end > start
            hi = bisect.bisect_right(token_ends, end)    # first token end > end
            positions = list(range(lo, hi))
            if not positions:
                continue  # word's chars fall inside a cross-word token: 0.0
            (head_words if positions[-1] < budget else tail_words).append(
                (row_idx, positions))

        if head_words:
            # Head is one variable-length window per passage: a batch of one.
            score_batch([(0, min(T, budget), head_words)])

        # Anchor each tail window to end at its word, keeping the most-recent
        # ``budget`` tokens -- but never start past the word's first target token,
        # so (like the per-word path) the whole target is always scored even if it
        # alone exceeds ``budget``. Windows of equal length share a bucket so each
        # batch stacks without padding; the ``budget``-length windows (all but the
        # rare over-long target) form one bucket batched ``batch_size`` at a time.
        buckets = defaultdict(list)  # window length -> [(a, b, [(row_idx, positions)])]
        for row_idx, positions in tail_words:
            b = positions[-1] + 1
            a = min(max(0, b - budget), positions[0])
            buckets[b - a].append((a, b, [(row_idx, positions)]))
        for specs in buckets.values():
            for i in range(0, len(specs), batch_size):
                score_batch(specs[i:i + batch_size])

    return result


def compute_surprisal(model_name, stims_df, device="cpu", backend="auto",
                      revision=None, trust_remote_code=True, batch_size=32):
    """Add a surprisal column for ``model_name`` over ``stims_df``.

    stims_df must contain a ``FullTextMarked`` column. Returns (df, column_name).
    ``batch_size`` bounds the causal path's tail-window batches (see
    :func:`_compute_causal_surprisal_by_passage`); it is ignored by other backends.
    """
    ilm, backend = load_scorer(model_name, backend, device, revision, trust_remote_code)
    max_len = model_max_len(ilm)
    col = surprisal_column_name(model_name)
    out = stims_df.copy()
    if backend == "causal":
        # Score a whole passage per forward pass instead of one pass per word.
        out[col] = _compute_causal_surprisal_by_passage(
            ilm, out, max_len, batch_size=batch_size)
    else:
        get_surprisal = make_get_surprisal(ilm, backend, max_len)
        tqdm.pandas(desc=f"{col} [{backend}]")
        out[col] = out["FullTextMarked"].progress_apply(get_surprisal)
    return out, col
