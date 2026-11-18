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
- **Stage 7 (rational-coordinate stability / probes)**: complete — 256-state
  coefficient stability audit, full 388-D per-output Ridge, PCA truncation
  (64/128/256/320/full), train-only PLS2, denominator/S11/S21 ablation,
  complex-plane probes, quadratic supervised inverse and residual structure.
  Coefficients are stable (median relative drift 1.9e-4) and PCA256≈full
  (physical MAE 0.02429); full coefficients alone (0.02430) do not beat
  Stage 6's rational+raw V4 (0.02329). PLS, probes and quadratic did not help;
  best Stage 7 one-shot is 19.83 dB mean miss (0% pass), so no canonical/Loewner
  transition is justified.
- **Stage 8 (learned forward / refinement gate)**: complete — raw-PCA (K=256/512)
  and rational+raw forward surrogates, 7000/1000 train split, surrogate-floor
  and Jacobian tests, Adam and LM/tanh refinement. The forward surrogate is
  not accurate enough (Z1 R² 0.02–0.06; Z2 R² 0.44), `r0/r_floor ≈ 1`, Jacobian
  quality is poor, and refinement leaves true physical MAE at ~0.0233 (no
  `<0.018` gate). No validation one-shot was run; no learned unrolling.
- **Stage 9 (smooth spec / zero-order residual policy)**: complete — smooth
  seven-spec objective calibrated against the real simulator (Spearman 0.954),
  coordinate-gradient checks (cosine stable across eps), one-step oracle
  (20.01→17.79 dB), SPSA fidelity (K=16 cosine 0.539), and a frozen-base
  residual head trained with simulator-in-the-loop SPSA. Full-8000 training
  validation: mean **14.66 dB**, median 14.83, P90 16.74 (from 21.56/19.06/33.69),
  but pass remains **0/1000**; both S11 requirements remain at 0% pass while
  S21 is largely fixed. No encoder unfreeze was run.
- **Stage 10 (S11-targeted gate)**: complete — clipping/saturation audit,
  exact per-requirement gradients, S11/S21 conflict analysis, S11 gradient SVD,
  full-41 vs 9-D B11 oracle, low-D SPSA and an S11-specific head. Stage 9 policy
  is heavily clipping-dependent (46.8% coordinates, 100% samples); S11 gradient is
  larger than S21 but less aligned with full-objective directions. Exact oracle
  reaches S11 13.4→8.10 dB / overall 9.34, but the S11 head only reaches
  S11 12.69 / pass 0, so no encoder unfreeze was run.
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
