from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


ROOT = Path(__file__).parents[1]
HANZI_DIR = ROOT / "tasks" / "hanzi"
YAML_LOADER = type("YamlLoader", (yaml.SafeLoader,), {})
YAML_LOADER.add_constructor(
    "!function", lambda loader, node: loader.construct_scalar(node)
)
SPEC = importlib.util.spec_from_file_location("hanzi_utils", HANZI_DIR / "utils.py")
HANZI_UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HANZI_UTILS)
COLLATE_SPEC = importlib.util.spec_from_file_location(
    "collate_results", ROOT / "scripts" / "collate_results.py"
)
COLLATE_RESULTS = importlib.util.module_from_spec(COLLATE_SPEC)
COLLATE_SPEC.loader.exec_module(COLLATE_RESULTS)


class DummyTokenizer:
    unk_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [0] if "UNK" in text else [1]


class HanziTaskTests(unittest.TestCase):
    def test_task_configs_use_expected_datasets_and_choices(self):
        expected = {
            "hanzi_structure": "chinese-babylm-org/hanzi-structure",
            "hanzi_pinyin": "chinese-babylm-org/hanzi-pinyin",
        }
        for task, dataset in expected.items():
            with (HANZI_DIR / f"{task}.yaml").open() as config_file:
                config = yaml.load(config_file, Loader=YAML_LOADER)
            self.assertEqual(config["task"], task)
            self.assertEqual(config["dataset_path"], dataset)
            self.assertEqual(config["doc_to_target"], 0)
            self.assertIn("sentence_good", config["doc_to_choice"])
            self.assertIn("sentence_bad", config["doc_to_choice"])

    def test_tasks_are_part_of_chinese_group_and_collation(self):
        with (
            ROOT / "tasks" / "zeroshot_babybabellm" / "zeroshot_zho.yaml"
        ).open() as group_file:
            group = yaml.safe_load(group_file)
        expected = {"hanzi_structure", "hanzi_pinyin"}
        self.assertTrue(expected.issubset(group["task"]))
        self.assertTrue(expected.issubset(COLLATE_RESULTS.EXPECTED_ZEROSHOT["zho"]))

    def test_hanzi_scores_are_excluded_from_score_submission(self):
        parsed = COLLATE_RESULTS.parse_zeroshot(
            {
                "hanzi_structure": {
                    "alias": "hanzi_structure",
                    "acc,none": 0.6,
                },
                "hanzi_pinyin": {
                    "alias": "hanzi_pinyin",
                    "acc,none": 0.5,
                },
            }
        )
        self.assertEqual(parsed, {})

    def test_hanzi_sample_is_reduced_to_raw_server_inputs(self):
        raw = COLLATE_RESULTS._hanzi_sample_prediction(
            {
                "filtered_resps": [["-1.0", "False"], ["-2.0", "False"]],
                "target": "0",
                "acc": 1.0,
                "doc_hash": "item-1",
            }
        )
        self.assertEqual(raw, {"scores": [-1.0, -2.0], "has_unk": False})

    def test_hanzi_unknown_status_is_recovered_without_copying_accuracy(self):
        raw = COLLATE_RESULTS._hanzi_sample_prediction(
            {
                "filtered_resps": [["-1.0", "False"], ["-2.0", "False"]],
                "target": "0",
                "acc": 0.0,
                "doc_hash": "item-1",
            }
        )
        self.assertEqual(raw, {"scores": [-1.0, -2.0], "has_unk": True})

    @patch.object(HANZI_UTILS, "_tokenizer", return_value=DummyTokenizer())
    def test_good_sentence_with_higher_likelihood_is_correct(self, _tokenizer):
        result = HANZI_UTILS.process_results(
            {"sentence_good": "good", "sentence_bad": "bad"},
            [(2.0, True), (1.0, False)],
        )
        self.assertEqual(result, {"acc": 1.0})

    @patch.object(HANZI_UTILS, "_tokenizer", return_value=DummyTokenizer())
    def test_bad_sentence_with_higher_likelihood_is_incorrect(self, _tokenizer):
        result = HANZI_UTILS.process_results(
            {"sentence_good": "good", "sentence_bad": "bad"},
            [(1.0, False), (2.0, True)],
        )
        self.assertEqual(result, {"acc": 0.0})

    @patch.object(HANZI_UTILS, "_tokenizer", return_value=DummyTokenizer())
    def test_item_with_unknown_token_is_incorrect(self, _tokenizer):
        result = HANZI_UTILS.process_results(
            {"sentence_good": "UNK", "sentence_bad": "bad"},
            [(2.0, True), (1.0, False)],
        )
        self.assertEqual(result, {"acc": 0.0})


if __name__ == "__main__":
    unittest.main()
