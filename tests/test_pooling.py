"""Pooling tests: shapes, local-softmax normalization, tau, gradients."""

from __future__ import annotations

import unittest

import torch

from _fixtures import ROOT  # noqa: F401

from solution.pooling import (
    AveragePool1D,
    InformationPreservingPool1D,
    MultiScaleAggregator,
    build_pooling,
)


def _num_windows(length: int, kernel: int, stride: int) -> int:
    return (length - kernel) // stride + 1


class TestInformationPreservingPool(unittest.TestCase):
    def test_output_shapes(self):
        for length, kernel, stride in ((2001, 4, 2), (199, 5, 3), (64, 8, 8), (201, 3, 1)):
            pool = InformationPreservingPool1D(
                in_channels=6, out_channels=10, kernel_size=kernel, stride=stride
            )
            x = torch.randn(2, 6, length)
            y = pool(x)
            expected = _num_windows(length, kernel, stride)
            self.assertEqual(tuple(y.shape), (2, 10, expected))

    def test_softmax_weights_sum_to_one(self):
        pool = InformationPreservingPool1D(4, 4, kernel_size=5, stride=2)
        x = torch.randn(3, 4, 101)
        plus, minus = pool.softmax_weights(x)
        self.assertEqual(tuple(plus.shape), (3, _num_windows(101, 5, 2), 5))
        self.assertTrue(torch.allclose(plus.sum(-1), torch.ones_like(plus.sum(-1)), atol=1e-5))
        self.assertTrue(torch.allclose(minus.sum(-1), torch.ones_like(minus.sum(-1)), atol=1e-5))

    def test_tau_is_positive(self):
        learnable = InformationPreservingPool1D(4, 4, tau=0.7, learnable_tau=True, tau_min=0.05)
        self.assertGreater(float(learnable.current_tau()), 0.0)
        fixed = InformationPreservingPool1D(4, 4, tau=0.7, learnable_tau=False, tau_min=0.05)
        self.assertGreater(float(fixed.current_tau()), 0.0)
        self.assertFalse(fixed.log_tau.requires_grad)

    def test_forward_backward_finite(self):
        pool = InformationPreservingPool1D(5, 7, kernel_size=4, stride=2)
        x = torch.randn(2, 5, 65, requires_grad=True)
        y = pool(x)
        self.assertTrue(bool(torch.isfinite(y).all()))
        y.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        for name, parameter in pool.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)

    def test_factory_and_average_reference(self):
        pool = build_pooling(
            {"type": "info_preserving", "out_channels": 9, "kernel_size": 4, "stride": 2}, 5
        )
        self.assertIsInstance(pool, InformationPreservingPool1D)
        x = torch.randn(1, 5, 33)
        self.assertEqual(tuple(pool(x).shape), (1, 9, _num_windows(33, 4, 2)))
        avg = AveragePool1D(5, 9, kernel_size=4, stride=2)
        self.assertEqual(tuple(avg(x).shape), (1, 9, _num_windows(33, 4, 2)))

    def test_multiscale_aggregator(self):
        features = [torch.randn(2, 8, 100), torch.randn(2, 12, 50), torch.randn(2, 16, 25)]
        aggregator = MultiScaleAggregator(mode="multiscale", length=8)
        out = aggregator(features)
        self.assertEqual(tuple(out.shape), (2, 36, 8))
        self.assertEqual(tuple(MultiScaleAggregator("last", 8)(features).shape), (2, 16, 8))
        self.assertEqual(tuple(MultiScaleAggregator("global", 8)(features).shape), (2, 36, 1))


if __name__ == "__main__":
    unittest.main()
