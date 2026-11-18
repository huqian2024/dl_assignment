# CM19 One-Shot S → ΔM

Deep-learning assignment solution for the anonymous CM19 dual-band two-port filter:
predict a one-shot 41-dimensional correction from the raw complex response.

```text
raw S (B, 4, 2001)  ->  delta_m (B, 41)
```

## Status

- **Stage 1 (engineering)**: complete — data cache, compliant anchor, DSP front-ends,
  composable ResNet/FNO/pooling/head, four routes, training/eval/export, 22 unit tests.
- **Stage 2 (Route 3 research)**: complete — delay representations R1/R2/R3,
  FNO modes 32/64/96/128, state-error and failure analysis.
- **Stage 3 (pairwise/observable control)**: complete — response PCA, local
  Jacobian spectrum, observable subspace, linear differential controllers.
  Result: no usable signal (0% pass, mean miss ≈20.6 dB), so no neural
  controller was trained.
- **Stage 4 (per-state teacher pilot)**: complete — anchor-line search and
  constrained local teacher search on 256 train states. The compliant region is
  a tiny pocket near the anchor; teacher corrections are ~97% of the anchor
  correction, dense, and no more learnable/consistent. No 8000-state expansion.
- **Stage 5 (system-identification features)**: complete — complex-algebra
  features, shared-denominator rational fitting (Chebyshev, frequency holdout)
  and Ridge/MLP regression. Rational coefficients improve parameter MAE
  0.0295→0.0250 and weak-parameter R² substantially, but one-shot remains
  0% pass / mean miss 20.5 dB.
- **Stage 6 (existing-8000 system-ID + ambiguity check)**: complete — shared-pole
  Vector Fitting pilot (unstable, rejected), order-64 coefficients on all 8000
  cached train states with group whitening, global/local Ridge and small ResMLP,
  1k/2k/4k/8k learning curve, exact raw-response nearest-neighbor and
  cross-correction tests. No new states/seeds. Best cheap model: Ridge on
  whitened order64 + raw PCA64, physical MAE **0.02329**, mean R² **0.564**;
  best one-shot is ResMLP at **18.96 dB** mean miss (still 0% pass). The closest
  raw-response pair has `d_S=0.258`; therefore no true near-collision was found
  and Stage 5's ambiguity interpretation is retracted.
- **Final reference model**: R3 raw+Tukey(α=0.25), FNO 32 modes, 3.28M parameters.
  Validation MAE `0.0322`, one-shot pass rate `0/1000` (current known limitation).

See **[HANDOVER.md](HANDOVER.md)** for the full handover document (environment, data,
commands, results, diagnostics, known issues and next steps).

## Quick start

```bash
/usr/bin/python3.10 -m venv .venv
.venv/bin/pip install numpy==1.24.4 torch==2.4.1 matplotlib==3.7.5

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/build_dataset.py --workers 12

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/find_anchor.py --workers 12 --population 256 --iterations 150

.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/smoke_test.py --tiny-steps 30
```

## Layout

| Path | Purpose |
|---|---|
| `assignments/` | Provided assignment materials (simulator, notebook, design CSVs) |
| `solution/` | Core library (data, DSP, blocks, encoders, pooling, heads, models, trainer, evaluator, export, analysis, pairwise control) |
| `configs/` | JSON route/experiment configurations |
| `scripts/` | Data, anchor, train, evaluate, export, analyze, smoke-test CLIs |
| `tests/` | Unit tests (`unittest`) |
| `HANDOVER.md` | Detailed handover document |

## Notes

- Public evaluation seeds `10000–10049` are never used; the evaluator rejects them.
- Training uses bf16 AMP with FFT forced to float32 (fp16 + odd-length cuFFT is unsupported).
- Caches, run outputs, model weights and generated `.npz` artifacts are git-ignored;
  see HANDOVER §4 for their local paths and regeneration commands.
