"""Export tests: TorchScript save/load, eager agreement, CPU execution."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from _fixtures import ROUTE_CONFIGS, TempArtifacts, ROOT  # noqa: F401

from solution.export import export_torchscript
from solution.models import build_model
from solution.simulator import load_simulator


class TestTorchScriptExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.simulator = load_simulator()
        cls._artifacts = TempArtifacts()
        cls._artifacts.__enter__()
        cls._tmp = tempfile.TemporaryDirectory(prefix="cm19_export_")
        cls.base = Path(cls._tmp.name)
        cls.results = {}
        cls.models = {}
        for path in ROUTE_CONFIGS:
            model = build_model(cls._artifacts.config(path), cls.simulator)
            name = Path(path).stem
            cls.models[name] = model
            cls.results[name] = export_torchscript(
                model, cls.simulator, output_dir=str(cls.base / name), seed=99999
            )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        cls._artifacts.__exit__(None, None, None)

    def test_artifacts_exist(self):
        for name, result in self.results.items():
            self.assertTrue(Path(result["model_path"]).exists(), name)
            metadata = json.loads(Path(result["metadata_path"]).read_text())
            self.assertEqual(metadata["input_shape"], [None, 4, 2001])
            self.assertEqual(metadata["output_shape"], [None, 41])
            self.assertEqual(
                metadata["channel_order"], ["re_s11", "im_s11", "re_s21", "im_s21"]
            )
            self.assertFalse(metadata["uses_simulator_in_forward"])
            self.assertLess(metadata["parameter_count"], 20_000_000)

    def test_numerical_agreement_and_cpu_load(self):
        for name, result in self.results.items():
            for batch, verification in result["verification"].items():
                self.assertLess(verification["max_abs_diff"], 1e-4, f"{name} B={batch}")
                self.assertEqual(verification["shape"], [int(batch), 41])
                self.assertTrue(verification["finite"])
            loaded = torch.jit.load(result["model_path"], map_location="cpu").eval()
            x = torch.randn(3, 4, 2001)
            with torch.no_grad():
                out = loaded(x)
            self.assertEqual(tuple(out.shape), (3, 41))
            self.assertEqual(out.dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(out).all()))


if __name__ == "__main__":
    unittest.main()
