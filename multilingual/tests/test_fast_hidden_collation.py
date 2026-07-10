from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "collate_results", ROOT / "scripts" / "collate_results.py"
)
COLLATE_RESULTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLATE_RESULTS)


class FastHiddenCollationTests(unittest.TestCase):
    def test_loads_hanzi_and_meco_from_the_requested_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_name = "test-model"
            revision = "chck_1M"

            hanzi_dir = root / "results" / revision / f"org__{model_name}"
            hanzi_dir.mkdir(parents=True)
            sample = {
                "doc_id": 7,
                "doc_hash": "hash-7",
                "filtered_resps": [[-1.0, False], [-2.0, False]],
                "target": 0,
                "acc": 1.0,
            }
            for task in ("hanzi_structure", "hanzi_pinyin"):
                (hanzi_dir / f"samples_{task}_2026-01-01.jsonl").write_text(
                    json.dumps(sample) + "\n"
                )

            meco_dir = (
                root
                / "meco"
                / "results"
                / revision
                / f"meco_org__{model_name}"
            )
            meco_dir.mkdir(parents=True)
            (meco_dir / "predictions_meco.json").write_text(
                json.dumps(
                    {
                        "meco_l1": {"zho": {"1:1": 8.42}},
                        "meco_l2": {"nld": {"1:1": 7.25}},
                    }
                )
            )

            hidden = COLLATE_RESULTS.load_hidden_predictions(
                root, model_name, revision
            )

            self.assertEqual(
                hidden["hanzi_structure"],
                {"7:hash-7": {"scores": [-1.0, -2.0], "has_unk": False}},
            )
            self.assertEqual(hidden["hanzi_pinyin"], hidden["hanzi_structure"])
            self.assertEqual(hidden["meco_l1"], {"zho": {"1:1": 8.42}})
            self.assertEqual(hidden["meco_l2"], {"nld": {"1:1": 7.25}})
            self.assertEqual(
                COLLATE_RESULTS.load_hidden_predictions(
                    root, model_name, "chck_2M"
                ),
                {},
            )

    def test_fast_payload_labels_every_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = COLLATE_RESULTS.load_fast_hidden_predictions(
                Path(tmp), "test-model", revisions=["chck_1M", "chck_2M"]
            )

        self.assertEqual(
            payload,
            [
                {"revision": "chck_1M", "hidden": {}},
                {"revision": "chck_2M", "hidden": {}},
            ],
        )


if __name__ == "__main__":
    unittest.main()
