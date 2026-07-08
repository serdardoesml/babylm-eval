from __future__ import annotations

import unittest
from unittest import mock

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

from finetune import pos_data

UPOS = ["NOUN", "VERB", "ADJ"]


def _lang_dataset(n: int, lang: str, names=UPOS) -> Dataset:
    features = Features({"tokens": Sequence(Value("string")),
                         "upos": Sequence(ClassLabel(names=names))})
    return Dataset.from_dict(
        {"tokens": [[f"{lang}{i}"] for i in range(n)],
         "upos": [[i % len(names)] for i in range(n)]},
        features=features,
    )


def _lang_splits(n: int, lang: str, names=UPOS) -> DatasetDict:
    return DatasetDict(
        train=_lang_dataset(n, lang, names),
        validation=_lang_dataset(max(2, n // 4), lang, names),
        test=_lang_dataset(max(2, n // 4), lang, names),
    )


class PosMixtureTests(unittest.TestCase):
    def _patch(self, per_lang):
        return mock.patch.object(
            pos_data, "build_pos_splits",
            side_effect=lambda configs, *a, **k: {lg: per_lang[lg] for lg in configs},
        )

    def test_single_language_is_passthrough(self):
        with self._patch({"en": _lang_splits(100, "en")}):
            out = pos_data.build_pos_mixture({"en": "en_ewt"})
        self.assertEqual(set(out), {"train", "validation", "test"})
        self.assertEqual(len(out["train"]), 100)

    def test_multi_language_train_is_uniform(self):
        per_lang = {"en": _lang_splits(100, "en"),
                    "nl": _lang_splits(40, "nl"),
                    "zh": _lang_splits(70, "zh")}
        with self._patch(per_lang):
            out = pos_data.build_pos_mixture({"en": "e", "nl": "n", "zh": "z"}, seed=1)
        # Each language capped to the per-language minimum (40) -> 3 * 40.
        self.assertEqual(len(out["train"]), 120)
        counts: dict[str, int] = {}
        for row in out["train"]["tokens"]:
            counts[row[0][:2]] = counts.get(row[0][:2], 0) + 1
        self.assertEqual(counts, {"en": 40, "nl": 40, "zh": 40})

    def test_multi_language_has_per_language_eval_splits(self):
        per_lang = {"en": _lang_splits(100, "en"), "nl": _lang_splits(40, "nl")}
        with self._patch(per_lang):
            out = pos_data.build_pos_mixture({"en": "e", "nl": "n"})
        self.assertEqual(
            set(out),
            {"train", "validation", "validation_en", "validation_nl", "test_en", "test_nl"},
        )
        # Pooled validation is the concatenation of the per-language dev splits.
        self.assertEqual(len(out["validation"]),
                         len(out["validation_en"]) + len(out["validation_nl"]))

    def test_mismatched_label_sets_raise(self):
        per_lang = {"en": _lang_splits(10, "en", UPOS),
                    "nl": _lang_splits(10, "nl", ["X", "Y"])}
        with self._patch(per_lang):
            with self.assertRaisesRegex(ValueError, "UPOS label set"):
                pos_data.build_pos_mixture({"en": "e", "nl": "n"})


if __name__ == "__main__":
    unittest.main()
