"""Differentiable, batch-first DSP front-ends for the raw CM19 response.

All transforms are pure torch and TorchScript-friendly: complex data are kept
as explicit real/imaginary channel pairs so no complex tensors ever enter the
exported graph.

Channel order of the raw response is fixed by the assignment:

    [Re(S11), Im(S11), Re(S21), Im(S21)]

Terminology: applying an inverse FFT to 2001 *sampled* frequency points yields
a band-limited delay representation, not the physical impulse response.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Complex FFT / IFFT built from real FFTs (scriptable, no complex dtypes)
# ---------------------------------------------------------------------------

def _real_full_fft(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-length complex FFT of a real signal, returned as (real, imag).

    Only odd signal lengths are supported, which covers the assignment's 2001
    frequency samples and avoids Nyquist-bin bookkeeping.
    """
    n = int(x.shape[-1])
    if n % 2 == 0:
        raise ValueError("Only odd signal lengths are supported by this helper")
    half = torch.fft.rfft(x, dim=-1)
    re_half = half.real
    im_half = half.imag
    re_mirror = re_half[..., 1:].flip(-1)
    im_mirror = -im_half[..., 1:].flip(-1)
    real = torch.cat([re_half, re_mirror], dim=-1)
    imag = torch.cat([im_half, im_mirror], dim=-1)
    return real, imag


def _no_autocast(device: torch.device):
    """cuFFT cannot run odd-length transforms in half precision under AMP."""
    return torch.autocast(device_type=device.type, enabled=False)


def _as_fft_float(x: torch.Tensor) -> torch.Tensor:
    """Promote half/bfloat16 to float32 for FFT; leave float32/float64 alone."""
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
    return x


def complex_fft(re: torch.Tensor, im: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FFT of ``re + j*im`` along the last axis, returned as (re, im)."""
    with _no_autocast(re.device):
        ar, ai = _real_full_fft(_as_fft_float(re))
        br, bi = _real_full_fft(_as_fft_float(im))
        return ar - bi, ai + br


def complex_ifft(re: torch.Tensor, im: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse FFT (1/N convention) of ``re + j*im`` along the last axis.

    Uses ``ifft(x) = conj(fft(conj(x))) / N`` so that only real FFTs are used
    internally, which keeps the operation TorchScript-friendly.
    """
    with _no_autocast(re.device):
        re_f = _as_fft_float(re)
        im_f = _as_fft_float(im)
        n = float(re_f.shape[-1])
        ar, ai = _real_full_fft(re_f)
        br, bi = _real_full_fft(im_f)
        return (ar + bi) / n, (br - ai) / n


def complex_ifft_channels(response: torch.Tensor) -> torch.Tensor:
    """(B,4,F) raw S -> (B,4,F) ``[Re h11, Im h11, Re h21, Im h21]``."""
    if response.dim() != 3 or response.shape[1] != 4:
        raise ValueError(f"Expected response (B,4,F), got {tuple(response.shape)}")
    h11 = complex_ifft(response[:, 0], response[:, 1])
    h21 = complex_ifft(response[:, 2], response[:, 3])
    return torch.stack([h11[0], h11[1], h21[0], h21[1]], dim=1)


# ---------------------------------------------------------------------------
# Deterministic windows
# ---------------------------------------------------------------------------

def tukey_window(length: int, alpha: float, dtype=torch.float32) -> torch.Tensor:
    """Symmetric Tukey window; ``alpha=0`` -> rectangular, ``alpha=1`` -> Hann."""
    alpha = float(min(max(alpha, 0.0), 1.0))
    if alpha <= 0.0:
        return torch.ones(length, dtype=dtype)
    if alpha >= 1.0:
        return torch.hann_window(length, periodic=False, dtype=dtype)
    positions = torch.linspace(0.0, 1.0, length, dtype=dtype)
    left = 0.5 * (1.0 + torch.cos(math.pi * (2.0 * positions / alpha - 1.0)))
    right = 0.5 * (
        1.0 + torch.cos(math.pi * (2.0 * positions / alpha - 2.0 / alpha + 1.0))
    )
    window = torch.where(positions < alpha / 2.0, left, torch.ones_like(positions))
    window = torch.where(positions >= 1.0 - alpha / 2.0, right, window)
    # Guard the exact endpoints against round-off leaving tiny negative values.
    return torch.clamp(window, min=0.0)


def make_window(name: str, length: int, tukey_alpha: float = 0.25) -> torch.Tensor:
    key = str(name).lower()
    if key in ("none", "rect", "rectangular", ""):
        return torch.ones(length, dtype=torch.float32)
    if key == "tukey":
        return tukey_window(length, tukey_alpha)
    if key == "hann":
        return tukey_window(length, 1.0)
    raise ValueError(f"Unknown window '{name}'")


def apply_window(response: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Multiply a complex (B,4,F) response by a real frequency window."""
    return response * window.view(1, 1, -1)


# ---------------------------------------------------------------------------
# Zero padding in the delay domain
# ---------------------------------------------------------------------------

def pad_delay(signal: torch.Tensor, left: int, right: int) -> torch.Tensor:
    if left < 0 or right < 0:
        raise ValueError("padding must be non-negative")
    if left == 0 and right == 0:
        return signal
    return F.pad(signal, (int(left), int(right)), mode="constant", value=0.0)


def crop_delay(signal: torch.Tensor, left: int, right: int) -> torch.Tensor:
    length = int(signal.shape[-1])
    end = length - int(right)
    if end <= int(left):
        raise ValueError("padding larger than the signal length")
    return signal[..., int(left):end]


# ---------------------------------------------------------------------------
# Representation modules
# ---------------------------------------------------------------------------

class FrequencyRepresentation(nn.Module):
    """Identity: keep the raw complex frequency response (4 channels)."""

    out_channels = 4

    def forward(self, response: torch.Tensor) -> torch.Tensor:
        return response


class DelayRepresentation(nn.Module):
    """Band-limited delay representation obtained by a complex IFFT.

    ``window`` selects the deterministic taper for the *windowed* branch; the
    un-windowed branch (``h0``) is always kept when ``concat_raw`` is true, so
    the default output is 8 channels:

        [Re h0_11, Im h0_11, Re h0_21, Im h0_21,
         Re hw_11, Im hw_11, Re hw_21, Im hw_21]
    """

    def __init__(
        self,
        num_freq_points: int,
        window: str = "tukey",
        tukey_alpha: float = 0.25,
        concat_raw: bool = True,
    ) -> None:
        super().__init__()
        self.num_freq_points = int(num_freq_points)
        self.window_name = str(window).lower()
        self.concat_raw = bool(concat_raw)
        taper = make_window(self.window_name, self.num_freq_points, tukey_alpha)
        self.register_buffer("taper", taper, persistent=True)
        has_window = self.window_name not in ("none", "rect", "rectangular", "")
        self.out_channels = 8 if (self.concat_raw and has_window) else 4

    def forward(self, response: torch.Tensor) -> torch.Tensor:
        raw_delay = complex_ifft_channels(response)
        if self.out_channels == 4:
            return raw_delay
        windowed = complex_ifft_channels(apply_window(response, self.taper))
        return torch.cat([raw_delay, windowed], dim=1)


def build_representation(rep_cfg: dict, num_freq_points: int) -> nn.Module:
    rep_type = str(rep_cfg.get("type", "frequency")).lower()
    if rep_type == "frequency":
        return FrequencyRepresentation()
    if rep_type == "delay":
        return DelayRepresentation(
            num_freq_points=int(num_freq_points),
            window=str(rep_cfg.get("window", "tukey")),
            tukey_alpha=float(rep_cfg.get("tukey_alpha", 0.25)),
            concat_raw=bool(rep_cfg.get("delay_concat_raw", True)),
        )
    raise ValueError(f"Unknown representation type '{rep_type}'")
