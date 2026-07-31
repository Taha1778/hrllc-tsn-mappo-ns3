"""Regression checks for centralized reference-suite baselines."""

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from paper_reference_full_suite import CPPOBaseline, DQNBaseline, SystemConfig


class CentralizedBaselineObservationTests(unittest.TestCase):
    """Ensure flattened baseline inputs match the environment observation shape."""

    def test_cpp_and_dqn_accept_all_supported_equipment_counts(self):
        for equipment_count in (2, 4, 8, 12):
            with self.subTest(equipment_count=equipment_count):
                config = SystemConfig(K_equipments=equipment_count)
                observations = np.zeros(
                    (equipment_count, equipment_count + 2), dtype=np.float32
                )

                for baseline_type in (CPPOBaseline, DQNBaseline):
                    actions = baseline_type(config, seed=1).get_actions(observations)
                    self.assertEqual(actions.shape, (equipment_count, 2))


if __name__ == "__main__":
    unittest.main()
