#!/usr/bin/env python3
"""Focused tests for motor stripe (face dependence) regression."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from pfm_stripe_removal import SURFACE_NORMAL_COMPONENT, masked_row_correlation, regress_out, remove_stripes  # noqa: E402


def synthetic_sheet(side: int = 30, spacing: float = 2.0):
    """A flat grid of vertices with 4-neighbor adjacency and normals whose y component alternates in bands."""
    gx, gy = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    coords = np.column_stack([gx.ravel() * spacing, gy.ravel() * spacing, np.zeros(side * side)])
    normals = np.zeros((side * side, 3))
    normals[:, SURFACE_NORMAL_COMPONENT] = np.sin(gy.ravel() * np.pi / 3)
    index = np.arange(side * side).reshape(side, side)
    r = np.concatenate([index[:-1].ravel(), index[:, :-1].ravel()])
    c = np.concatenate([index[1:].ravel(), index[:, 1:].ravel()])
    adjacency = sp.csr_matrix((np.ones(2 * r.size, dtype=bool), (np.r_[r, c], np.r_[c, r])), shape=(side * side,) * 2)
    return coords, normals, adjacency


class StripeRemovalTests(unittest.TestCase):
    def test_masked_row_correlation_matches_corrcoef(self):
        rng = np.random.default_rng(0)
        Y, x = rng.normal(size=(5, 40)), rng.normal(size=40)
        mask = rng.random((5, 40)) > 0.3
        expected = [np.corrcoef(Y[i, mask[i]], x[mask[i]])[0, 1] for i in range(5)]
        np.testing.assert_allclose(masked_row_correlation(Y, x, mask), expected, atol=1e-10)

    def test_regress_out_removes_regressor_and_mean(self):
        rng = np.random.default_rng(1)
        reg = rng.normal(size=(200, 1))
        data = (3 * reg + 5 + 0.01 * rng.normal(size=(200, 4))).astype(np.float32)
        residual = regress_out(data, reg)
        self.assertLess(np.abs(residual).max(), 0.1)
        np.testing.assert_allclose(residual.mean(axis=0), 0, atol=1e-5)

    def test_synthetic_stripe_is_found_and_regressed(self):
        rng = np.random.default_rng(2)
        coords, normals, adjacency = synthetic_sheet()
        n_cort, n_time, n_sub = coords.shape[0], 300, 50
        data = rng.normal(size=(n_time, n_cort + n_sub)).astype(np.float32)
        stripe = rng.normal(size=n_time)
        data[:, :n_cort] += 2 * np.outer(stripe, normals[:, SURFACE_NORMAL_COMPONENT]).astype(np.float32)
        cs_roi = np.ones(n_cort, dtype=bool)
        dorsal_roi = np.zeros(n_cort, dtype=bool)
        dorsal_roi[: n_cort // 4] = True

        regressed, before, after, summary, table = remove_stripes(data, coords, normals, adjacency, cs_roi, dorsal_roi)
        self.assertGreaterEqual(summary["components_regressed"], 1)
        self.assertGreater(summary["stripiness_before"], 0.5)
        self.assertLess(summary["stripiness_after"], summary["stripiness_before"] / 2)
        self.assertEqual(regressed.shape, data.shape)
        self.assertEqual(before.shape, (n_cort,))
        self.assertEqual(len(table), summary["components"])


if __name__ == "__main__":
    unittest.main()
