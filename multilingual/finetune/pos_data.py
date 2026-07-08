"""
Build POS-tagging train/dev/test splits from Universal Dependencies treebanks.

Pipeline per treebank:
  1. Load every split the treebank ships with (UD's native train/dev/test sizes
     vary and don't follow 80/10/10), and pool them.
  2. Keep only the columns needed for POS tagging (tokens + upos by default).
  3. Shuffle, then subsample to `max_sentences` if the pool is larger.
  4. Cut a fresh 80/10/10 train / validation / test split.

The core logic lives in `make_pos_split`, which operates on an in-memory
dataset so it can be unit-tested without network access. `build_pos_split`
is the thin I/O wrapper that loads from the Hub first.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Union

from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
)

RawData = Union[Dataset, DatasetDict]


def make_pos_split(
    raw: RawData,
    max_sentences: int | None = None,
    *,
    keep_columns: tuple[str, ...] = ("tokens", "upos"),
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> DatasetDict:
    """Pool, subsample, and 80/10/10-split an already-loaded treebank.

    Args:
        raw: A single Dataset, or a DatasetDict whose splits get concatenated.
        max_sentences: If set and the pool is larger, randomly keep this many
            sentences (sampling happens before the split, so the split is taken
            from the subsample).
        keep_columns: Columns to retain. ClassLabel features (e.g. `upos`) are
            preserved, so label names/ids survive.
        split_ratios: (train, dev, test); must sum to 1.0.
        seed: Controls shuffling, sampling, and the split for reproducibility.

    Returns:
        DatasetDict with `train`, `validation`, and `test` keys.
    """
    if abs(sum(split_ratios) - 1.0) > 1e-9:
        raise ValueError(f"split_ratios must sum to 1.0, got {split_ratios}")
    train_frac, dev_frac, test_frac = split_ratios

    # 1. Pool all incoming splits into one dataset.
    if isinstance(raw, DatasetDict):
        pooled = concatenate_datasets([raw[s] for s in raw.keys()])
    else:
        pooled = raw

    # 2. Keep only POS-relevant columns.
    missing = [c for c in keep_columns if c not in pooled.column_names]
    if missing:
        raise KeyError(f"missing expected columns {missing}; have {pooled.column_names}")
    pooled = pooled.select_columns(list(keep_columns))

    # 3. Shuffle, then subsample if needed (random because of the shuffle).
    pooled = pooled.shuffle(seed=seed)
    if max_sentences is not None and len(pooled) > max_sentences:
        pooled = pooled.select(range(max_sentences))

    if len(pooled) < 3:
        raise ValueError(f"need at least 3 sentences to split, got {len(pooled)}")

    # 4. 80/10/10. train_test_split is 2-way, so split twice: peel off train,
    #    then divide the holdout into dev/test proportionally.
    holdout_frac = dev_frac + test_frac
    first = pooled.train_test_split(test_size=holdout_frac, seed=seed)
    test_within_holdout = test_frac / holdout_frac
    second = first["test"].train_test_split(test_size=test_within_holdout, seed=seed)

    return DatasetDict(
        train=first["train"],
        validation=second["train"],
        test=second["test"],
    )


def build_pos_split(
    config: str,
    max_sentences: int | None = None,
    *,
    dataset_name: str = "universal-dependencies/universal_dependencies",
    keep_columns: tuple[str, ...] = ("tokens", "upos"),
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    trust_remote_code: bool = True,
) -> DatasetDict:
    """Load a UD treebank config from the Hub and build its 80/10/10 split.

    Example configs: "zh_gsd" (Chinese), "nl_alpino" (Dutch), "en_ewt" (English).
    If you hit a remote-code prompt, the parquet re-host "commul/universal_dependencies"
    exposes the same configs and loads without `trust_remote_code`.
    """
    raw = load_dataset(dataset_name, config, trust_remote_code=trust_remote_code)
    return make_pos_split(
        raw,
        max_sentences,
        keep_columns=keep_columns,
        split_ratios=split_ratios,
        seed=seed,
    )


def build_pos_splits(
    configs: Mapping[str, str] | Iterable[str],
    max_sentences: int | None = None,
    **kwargs,
) -> dict[str, DatasetDict]:
    """Build splits for several treebanks at once.

    `configs` may be a mapping of label -> config (e.g. {"zh": "zh_gsd"}) or a
    plain list of configs (the config name is used as the key).
    """
    if isinstance(configs, Mapping):
        items = configs.items()
    else:
        items = ((c, c) for c in configs)
    return {label: build_pos_split(cfg, max_sentences, **kwargs) for label, cfg in items}


def build_pos_mixture(
    configs: Mapping[str, str] | Iterable[str],
    max_sentences: int | None = None,
    *,
    seed: int = 42,
    uniform: bool = True,
    **kwargs,
) -> DatasetDict:
    """Build one cross-lingual POS split from one or more UD treebanks.

    Each treebank is split individually with `build_pos_split` (so every language
    gets its own reproducible 80/10/10 split), then the per-language splits are
    combined into a single DatasetDict for a joint (cross-lingual) finetune:

      * train: a uniform mixture -- when `uniform` (the default) every language
        contributes the same number of sentences (the per-language minimum), so
        no single treebank dominates; the pooled train set is then shuffled.
      * validation: the concatenation of every language's dev split (used for
        early stopping on the mixed metric).
      * validation_<lang> / test_<lang>: each language's own dev/test split,
        kept separate so the joint model can be scored per language.

    With a single treebank this is exactly its `build_pos_split` output, so the
    monolingual path is unchanged. All treebanks must share the same UPOS label
    set (UD guarantees this; a mismatch raises rather than failing cryptically).
    """
    per_lang = build_pos_splits(configs, max_sentences, seed=seed, **kwargs)
    if len(per_lang) == 1:
        return next(iter(per_lang.values()))

    def upos_names(ds: DatasetDict) -> list[str]:
        return ds["train"].features["upos"].feature.names

    reference = upos_names(next(iter(per_lang.values())))
    for label, ds in per_lang.items():
        if upos_names(ds) != reference:
            raise ValueError(
                f"treebank '{label}' has a different UPOS label set; cannot mix "
                "languages with incompatible tag inventories."
            )

    cap = min(len(ds["train"]) for ds in per_lang.values()) if uniform else None
    train_parts = []
    for ds in per_lang.values():
        part = ds["train"]
        if cap is not None and len(part) > cap:
            part = part.shuffle(seed=seed).select(range(cap))
        train_parts.append(part)

    splits = {
        "train": concatenate_datasets(train_parts).shuffle(seed=seed),
        "validation": concatenate_datasets([ds["validation"] for ds in per_lang.values()]),
    }
    for label, ds in per_lang.items():
        splits[f"validation_{label}"] = ds["validation"]
        splits[f"test_{label}"] = ds["test"]
    return DatasetDict(splits)


# Default UD treebank per language code, matching the multilingual finetune set.
DEFAULT_UD_CONFIGS = {
    "zh": "zh_gsdsimp",
    "nl": "nl_alpino",
    "en": "en_ewt",
}
