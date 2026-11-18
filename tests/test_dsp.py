"""DSP tests: complex IFFT/FFT correctness, windows, padding, channels."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from _fixtures import ROOT  # noqa: F401  (adds project root to sys.path)

from solution.dsp import (
    DelayRepresentation,
    complex_fft,
    complex_ifft,
    complex_ifft_channels,
    crop_delay,
    pad_delay,
    tukey_window,
)


class TestComplexTransform(unittest.TestCase):
    def test_ifft_matches_numpy(self):
        torch.manual_seed(0)
        re = torch.randn(3, 2001, dtype=torch.float64)
        im = torch.randn(3, 2001, dtype=torch.float64)
        out_re, out_im = complex_ifft(re, im)
        reference = np.fft.ifft(re.numpy() + 1j * im.numpy(), axis=-1)
        self.assertLess(float(np.abs(out_re.numpy() - reference.real).max()), 1e-10)
        self.assertLess(float(np.abs(out_im.numpy() - reference.imag).max()), 1e-10)

    def test_round_trip(self):
        torch.manual_seed(1)
        re = torch.randn(2, 2001)
        im = torch.randn(2, 2001)
        h_re, h_im = complex_ifft(re, im)
        r_re, r_im = complex_fft(h_re, h_im)
        self.assertLess(float((r_re - re).abs().max()), 1e-4)
        self.assertLess(float((r_im - im).abs().max()), 1e-4)

    def test_channel_order(self):
        response = torch.zeros(2, 4, 2001)
        response[:, 0, :] = 1.0  # Re(S11) = 1 -> h11 real impulse at delay 0
        delay = complex_ifft_channels(response)
        self.assertEqual(tuple(delay.shape), (2, 4, 2001))
        self.assertTrue(torch.allclose(delay[:, 0, 0], torch.ones(2), atol=1e-5))
        self.assertLess(float(delay[:, 0, 1:].abs().max()), 1e-6)
        self.assertLess(float(delay[:, 1:].abs().max()), 1e-6)

    def test_window(self):
        window = tukey_window(2001, 0.25)
        self.assertEqual(tuple(window.shape), (2001,))
        self.assertGreaterEqual(float(window.min()), 0.0)
        self.assertLessEqual(float(window.max()), 1.0 + 1e-6)
        self.assertAlmostEqual(float(window[0]), 0.0, places=5)
        self.assertAlmostEqual(float(window[-1]), 0.0, places=5)
        self.assertAlmostEqual(float(window[1000]), 1.0, places=5)
        rectangle = tukey_window(2001, 0.0)
        self.assertTrue(torch.allclose(rectangle, torch.ones(2001)))

    def test_padding_round_trip(self):
        x = torch.randn(2, 3, 201)
        padded = pad_delay(x, 7, 11)
        self.assertEqual(tuple(padded.shape), (2, 3, 219))
        restored = crop_delay(padded, 7, 11)
        self.assertTrue(torch.equal(restored, x))

    def test_delay_representation_shapes(self):
        response = torch.randn(2, 4, 2001)
        raw_only = DelayRepresentation(2001, window="none", concat_raw=True)
        self.assertEqual(raw_only.out_channels, 4)
        self.assertEqual(tuple(raw_only(response).shape), (2, 4, 2001))
        both = DelayRepresentation(2001, window="tukey", tukey_alpha=0.25, concat_raw=True)
        self.assertEqual(both.out_channels, 8)
        out = both(response)
        self.assertEqual(tuple(out.shape), (2, 8, 2001))
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_delay_batch_consistency(self):
        response = torch.randn(3, 4, 2001)
        module = DelayRepresentation(2001, window="hann", concat_raw=True)
        together = module(response)
        for idx in range(3):
            single = module(response[idx:idx + 1])
            self.assertLess(float((together[idx:idx + 1] - single).abs().max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
