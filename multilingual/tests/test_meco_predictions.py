from __future__ import annotations

import unittest

import pandas as pd

from meco.meco_py.cli import _prediction_map


class MecoPredictionTests(unittest.TestCase):
    def test_prediction_map_uses_stable_item_position_ids(self):
        frame = pd.DataFrame(
            {
                "itemid": [1.0, "2"],
                "wordnum": [3, 4.0],
                "model__score": [5.25, 6.5],
            }
        )
        self.assertEqual(
            _prediction_map(frame, "model__score"),
            {"1:3": 5.25, "2:4": 6.5},
        )

    def test_prediction_map_keeps_first_of_duplicate_ids(self):
        # Some MECO items share an (itemid, wordnum) position across hyphenation
        # variants; the id scheme collapses them, so we keep the first occurrence.
        frame = pd.DataFrame(
            {
                "itemid": [1, 1, 2],
                "wordnum": [139, 139, 5],
                "model": [3.0, 9.0, 4.0],
            }
        )
        self.assertEqual(
            _prediction_map(frame, "model"),
            {"1:139": 3.0, "2:5": 4.0},
        )

    def test_prediction_map_rejects_non_finite_values(self):
        frame = pd.DataFrame(
            {
                "itemid": [1],
                "wordnum": [1],
                "model": [float("nan")],
            }
        )
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            _prediction_map(frame, "model")


if __name__ == "__main__":
    unittest.main()
