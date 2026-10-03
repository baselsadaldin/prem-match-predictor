"""Regression checks for probability validity and betting row alignment."""
import unittest

import numpy as np
import pandas as pd

from betting import place_bets
from dixon_coles import outcome_probabilities


class ProbabilityTests(unittest.TestCase):
    def test_extreme_means_have_valid_probabilities(self):
        for rho in [-0.2, 0.2]:
            p = outcome_probabilities(
                np.array([0.2, 8., 2.]), np.array([8., 8., 0.2]), rho
            )
            self.assertTrue(np.isfinite(p).all())
            self.assertTrue((p >= 0).all())
            np.testing.assert_allclose(p.sum(axis=1), 1)

    def test_misaligned_bets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same index"):
            place_bets(pd.DataFrame(index=[1, 2]), pd.DataFrame(index=[2, 1]), [], 0)


if __name__ == "__main__":
    unittest.main()
