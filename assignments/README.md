# CM19 One-Shot S→ΔM Course Assignment

This directory contains:

- `s_to_delta_m_homework.ipynb`: the student-facing assignment notebook;
- `cm19_simulator.py`: the public simulator and compliance-checking interface;
- `envs/`: the CM19 numerical backend and design-loading module, including
  the three CM19 matrix/bounds CSV files in `envs/designs/`;
- `utils/`: the matrix-vector conversion utilities; and
- `requirements.txt`: dependencies for the simulator and student notebook.

This directory is self-contained. It can be copied or distributed on its own;
no modules or data from the parent repository are required. The bundled design
assets cover CM19 only.

No dataset, generated targets, reference solver, loss function, or reference network is included. Students receive the simulator and seven compliance requirements and must design the remaining solution pipeline themselves. The required final artifact is a standalone one-shot mapping:

```text
raw complex S response (4 × 2001) -> delta M (41)
```

## Running the Assignment

Use Python 3.9–3.11 (Python 3.10/3.11 recommended).

On Windows PowerShell:

```powershell
cd assignments  # omit this if already inside the distributed assignment directory
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter lab s_to_delta_m_homework.ipynb
```

On macOS or Linux:

```bash
cd assignments  # omit this if already inside the distributed assignment directory
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyter lab s_to_delta_m_homework.ipynb
```

Select the virtual environment as the notebook kernel. CPU execution is supported;
a CUDA-enabled PyTorch installation is optional for training.

The notebook also works when launched from the parent directory. Use the same
virtual environment's Python executable and pass
`assignments/s_to_delta_m_homework.ipynb` as the JupyterLab notebook path.

Both modes use the copies inside this directory, without requiring the repository
root on `PYTHONPATH`. Student TODO cells intentionally raise `NotImplementedError`
until the student implements them.

## Student-Facing Simulator Contract

`CM19Simulator` exposes only the assignment operations:

- `sample_state(seed)` generates a reproducible detuned state and response;
- `simulate(m_vector)` evaluates a legal 41-dimensional state;
- `check_response(response)` checks every published requirement;
- `apply_correction(current_vector, delta_m)` applies and evaluates one correction;
- `lower_bounds` and `upper_bounds` give the legal parameter range.

The final model receives only the raw response. It must not receive the current matrix vector or seed and must not call the simulator during inference.
