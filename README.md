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
| `solution/` | Core library (data, DSP, blocks, encoders, pooling, heads, models, trainer, evaluator, export, analysis) |
| `configs/` | JSON route/experiment configurations |
| `scripts/` | Data, anchor, train, evaluate, export, analyze, smoke-test CLIs |
| `tests/` | Unit tests (`unittest`) |
| `HANDOVER.md` | Detailed handover document |

## Notes

- Public evaluation seeds `10000–10049` are never used; the evaluator rejects them.
- Training uses bf16 AMP with FFT forced to float32 (fp16 + odd-length cuFFT is unsupported).
- Caches, run outputs, model weights and generated `.npz` artifacts are git-ignored;
  see HANDOVER §4 for their local paths and regeneration commands.
