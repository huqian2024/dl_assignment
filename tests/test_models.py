"""Model tests: four routes instantiate, forward/backward, parameter budget."""

from __future__ import annotations

import unittest

import torch

from _fixtures import ROUTE_CONFIGS, TempArtifacts, ROOT  # noqa: F401

from solution.encoders import SpectralConv1D
from solution.models import build_model
from solution.simulator import load_simulator
from solution.utils import count_parameters


class TestFNOBlocks(unittest.TestCase):
    def test_spectral_modes_bounds(self):
        block = SpectralConv1D(4, 6, n_modes=5000)
        x = torch.randn(2, 4, 129, requires_grad=True)
        y = block(x)
        self.assertEqual(tuple(y.shape), (2, 6, 129))
        self.assertTrue(bool(torch.isfinite(y).all()))
        y.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertTrue(bool(torch.isfinite(block.weight_re.grad).all()))


class TestRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.simulator = load_simulator()
        cls._artifacts = TempArtifacts()
        cls._artifacts.__enter__()
        cls.models = {
            path: build_model(cls._artifacts.config(path), cls.simulator).eval()
            for path in ROUTE_CONFIGS
        }
        torch.manual_seed(7)
        cls.responses = {
            1: torch.randn(1, 4, 2001),
            4: torch.randn(4, 4, 2001),
        }

    @classmethod
    def tearDownClass(cls):
        cls._artifacts.__exit__(None, None, None)

    def test_parameter_budget(self):
        for path, model in self.models.items():
            params = count_parameters(model)
            self.assertLess(params, 20_000_000, path)
            self.assertLess(params, 12_000_000, f"{path} exceeds the engineering target")
            print(f"\n{path}: {params:,} trainable parameters")

    def test_forward_shapes_and_finiteness(self):
        for path, model in self.models.items():
            for batch, x in self.responses.items():
                with torch.no_grad():
                    out = model(x)
                self.assertEqual(tuple(out.shape), (batch, 41), path)
                self.assertEqual(out.dtype, torch.float32, path)
                self.assertTrue(bool(torch.isfinite(out).all()), path)

    def test_state_algebra_consistency(self):
        for path, model in self.models.items():
            with torch.no_grad():
                u, m_hat, delta = model.forward_state(self.responses[1])
            center = model.state_normalizer.center
            half = model.state_normalizer.half_range
            self.assertTrue(torch.allclose(m_hat, center + half * u, atol=1e-5), path)
            self.assertTrue(torch.allclose(delta, model.anchor - m_hat, atol=1e-5), path)

    def test_backward_finite(self):
        for path, model in self.models.items():
            model.train()
            out = model(self.responses[1])
            out.square().mean().backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()), f"{path}:{name}")
            model.zero_grad(set_to_none=True)
            model.eval()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_forward_backward(self):
        for path, model in self.models.items():
            device = torch.device("cuda")
            gpu_model = model.to(device)
            x = self.responses[1].to(device)
            out = gpu_model(x)
            self.assertTrue(bool(torch.isfinite(out).all()), path)
            out.square().mean().backward()
            gpu_model.to("cpu")
            gpu_model.zero_grad(set_to_none=True)


class TestPublicSeedGuard(unittest.TestCase):
    def test_guard_rejects_public_seeds(self):
        from solution.evaluator import assert_not_public_seeds

        assert_not_public_seeds([0, 1, 20000])
        with self.assertRaises(ValueError):
            assert_not_public_seeds([10000])
        with self.assertRaises(ValueError):
            assert_not_public_seeds([20000, 10049])


if __name__ == "__main__":
    unittest.main()
