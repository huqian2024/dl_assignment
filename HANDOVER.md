# CM19 One-Shot S→ΔM — 工程交接文档

> 最后更新：2026-11-17  
> 项目根目录：`/home/qian/DL_assignment_2026`  
> 上游仓库：<https://github.com/huqian2024/dl_assignment.git>

---

## 0. 当前状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| Stage 1 | 工程准备：环境、数据缓存、anchor、DSP、FNO/ResNet/Pooling/Head、四条路线、export、smoke test | ✅ 完成 |
| Stage 2 | Route 3 正式研究：representation R1/R2/R3、FNO modes 32/64/96/128、state/one-shot/residual 分析 | ✅ 完成 |
| Stage 3 | Pairwise / observable-subspace control 研究：response PCA、local Jacobian、可观测子空间、线性差分 controller | ✅ 完成（线性无信号，按规则停止） |
| Stage 4 | Per-state compliant teacher 结构验证（256-state train pilot，线搜索 + 局部可行点搜索 + 几何/可学习性分析） | ✅ 完成（无积极信号，不扩展到 8000） |
| Stage 5 | System-identification / rational representation（complex algebra、shared-denominator fit、Ridge/MLP） | ✅ 完成（表示层有正信号，但 one-shot 仍 0；按规则不扩到 8000） |

**最终参考模型**：Route 3（Delay-FNO）+ **R3 raw+Tukey(α=0.25) + 32 modes**  
- 3,275,325 参数；
- validation MAE **0.03222**，P95 **0.08328**；
- one-shot pass rate **0/1000**（validation 与 internal test 均为 0）；
- 结论：state 估计精度比 one-shot 所需精度（单参数约 0.001–0.002）差约 20–30 倍，主要限制是数据规模/泛化与 41 维逆问题病态性，而不是网络容量。

---

## 1. 环境

```text
OS              Linux
Python          3.10.12（/.venv，使用 /usr/bin/python3.10 创建）
numpy           1.24.4
torch           2.4.1+cu121
matplotlib      3.7.5
GPU             NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, sm_89)
CUDA            torch CUDA runtime 12.1；torch.cuda.is_available() == True
```

安装（在新机器上）：

```bash
cd /home/qian/DL_assignment_2026
/usr/bin/python3.10 -m venv .venv
.venv/bin/pip install numpy==1.24.4 torch==2.4.1 matplotlib==3.7.5
```

说明：

- 训练默认 **bf16 AMP**。fp16 在奇数长度 cuFFT（2001/2129）上不受支持，且部分路线会溢出；FFT 已在模型内部强制 float32 并局部关闭 autocast。
- 使用多进程跑 simulator（数据生成、anchor 搜索）时，必须设置单线程 BLAS，否则会线程超订、几乎停滞：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 <command>
```

- 本机访问 GitHub 需要代理 `http://127.0.0.1:7890`（`https_proxy`/`http_proxy`）。

---

## 2. 目录结构

```text
assignments/                 作业原始材料（simulator、notebook、designs；不要修改）
solution/                    核心库（可复用、模块化）
  config.py                  JSON 配置加载 + 默认值
  utils.py                   路径/种子/JSON/计时/参数统计
  simulator.py               公开 simulator 封装、signed margins
  data.py                    数据缓存、manifest、完整性校验、norm stats
  normalization.py           通道/位置归一化、state 归一化、统计持久化
  dsp.py                     可微 complex FFT/IFFT、Tukey/Hann、padding/crop
  blocks.py                  ResBlock1D、ResFCBlock、norm 工厂
  pooling.py                 InformationPreservingPool1D、multiscale aggregator
  encoders.py                ResNetEncoder、SpectralConv1D、FNOConvResidualBlock、FNOEncoder
  heads.py                   ResMLPHead
  models.py                  组合式 OneShotStateModel（representation×encoder×pooling×head）
  anchor_search.py           CEM + local refinement，输出 compliant anchor
  losses.py                  Huber/MSE/L1 + 完整 physical error metrics
  trainer.py                 AdamW/AMP/cosine/early stopping/多准则 checkpoint/周期 one-shot 评估
  evaluator.py               官方 one-shot 评估、identification 评估、画图
  export.py                  TorchScript trace/save/load/一致性验证
  analysis.py                residual 数据集、失败分析、诊断图
configs/                     路线与实验配置（JSON）
scripts/                     数据/ anchor / 训练 / 评估 / 导出 / smoke / 分析 CLI
tests/                       unittest（DSP、pooling、models、export、seed guard）
HANDOVER.md                  本文件
```

**不在 git 中**（见 `.gitignore`）：`.venv/`、`cache/`、`runs/`、`*.npz`、`*.pt`、`submission_artifacts/`。  
其中 `artifacts/analysis/**/*.json` 与 `*.png` 会提交，用于保留实验结果摘要；残差 `.npz`、anchor 向量、norm stats、模型权重不提交（涉及作业学术诚信，见 §9）。

---

## 3. 作业 contract（必须遵守）

```text
输入       raw response, shape (B, 4, 2001)
通道顺序   [Re S11, Im S11, Re S21, Im S21]
输出       delta_m, shape (B, 41)，原始 matrix-parameter scale
forward    禁止调用 simulator；禁止以 seed / m_current 作为输入
归一化     所有统计、window、anchor 必须存为 register_buffer（导出后无需外部文件）
参数量     < 20,000,000（本项目目标 ≤ 10M）
导出       standalone TorchScript，CPU 可加载
public seeds 10000–10049 严禁用于训练/target/归一化/选模/调参
```

公开 API（`assignments/cm19_simulator.py`）：

```text
sample_state(seed) -> (m_current, response)
simulate(m_vector) -> response
check_response(response) -> dict
apply_correction(current_vector, delta_m) -> CorrectionResult
lower_bounds / upper_bounds
```

---

## 4. 数据与关键产物

### 4.1 数据 split（已固定）

```text
train          0 .. 7999       8000 states
validation     20000 .. 20999  1000 states
internal-test  30000 .. 30999  1000 states
public eval    10000 .. 10049  从未使用
```

缓存（本地，不提交）：

```text
cache/dataset/manifest.json
cache/dataset/{split}_responses.npy   float32 (N,4,2001)
cache/dataset/{split}_m_current.npy   float32 (N,41)
cache/dataset/{split}_seeds.npy       int64
```

生成结果：10,000 simulator calls，约 52.7 s（12 workers，~190 states/s），缓存 306.97 MB；manifest 记录 sha256、split 范围、calls、wall time。

### 4.2 Anchor（本地，不提交）

```text
artifacts/common/anchor.npz
artifacts/common/anchor.json
artifacts/common/anchor.search_config.json
```

```text
compliant              yes（公共 API 独立复核 7/7 通过）
min signed margin      +0.638683 dB
margins                s11_1 +1.0402, s11_2 +0.9593, s21_1 +0.8181,
                       s21_2 +0.6390, s21_3 +0.6387, s21_4 +0.6602, s21_5 +0.6471
search                 CEM (pop 256 × 150 iters) + local refine (60 steps)
simulator calls        40,322（含最终验证）
wall time              227.1 s
```

### 4.3 Normalization stats（本地，不提交）

`artifacts/common/norm_stats.npz`，只由 **train split** 估计，包含 4 个 key：

```text
freq                  raw S，4 通道
delay_none            raw delay，4 通道
delay_tukey           Tukey delay only，4 通道
delay_tukey_concat    raw + Tukey delay，8 通道（R3 使用）
```

每个 key 同时保存 `per_channel_*` 与 `position_*` 两套 mean/std。

### 4.4 最终模型的 residual 数据集（本地，不提交）

```text
artifacts/analysis/final_m32/validation/residuals.npz
artifacts/analysis/final_m32/validation/summary.json
artifacts/analysis/final_m32/validation/{per_parameter_error,error_vs_miss,max_miss_histogram}.png
artifacts/analysis/final_m32/validation/identifiability.json
artifacts/analysis/final_m32/validation/failure_drivers.json

artifacts/analysis/final_m32/internal_test/...（同上）
```

`residuals.npz` 字段：

```text
seeds, m_true, m_pred, error=m_pred-m_true, delta, corrected,
margins (N,7), requirement_names, passed, max_miss_db, clipped_entry_rate
```

可直接用于 parameter sensitivity、weighted loss、robust anchor、failure clustering、Dual-domain 设计。

### 4.5 重新生成命令

```bash
# 1) 数据 + norm stats（已存在时会跳过数据生成）
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/build_dataset.py --workers 12 \
  --norm-configs configs/route_delay_fno_r1_raw.json \
                 configs/route_delay_fno_r2_tukey.json \
                 configs/route_delay_fno_r3_raw_tukey.json

# 2) anchor（约 4 分钟）
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/find_anchor.py --workers 12 \
  --population 256 --iterations 150 --local-refine-steps 60 --early-stop-margin 2.0

# 3) 单测 / smoke
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/smoke_test.py --tiny-steps 30
```

---

## 5. 训练 / 评估 / 导出 / 分析

### 5.1 正式训练（Stage 2 实际使用）

```bash
# Representation study（R1 / R2 / R3），单卡顺序执行
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash -c '
for name in r1_raw r2_tukey r3_raw_tukey; do
  .venv/bin/python scripts/train.py \
    --config configs/route_delay_fno_${name}.json \
    --run-name ${name}_s2026 --output-dir runs/route3_repr --epochs 120
done'

# FNO mode study（32 / 96 / 128；64 复用 R3 的 run）
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash -c '
for m in 32 96 128; do
  .venv/bin/python scripts/train.py \
    --config configs/route_delay_fno_r3_m${m}.json \
    --run-name r3_m${m}_s2026 --output-dir runs/route3_modes --epochs 120
done'

# 诊断：aggregation length 64 / m32 第二 seed
.venv/bin/python scripts/train.py --config configs/route_delay_fno_r3_agg64.json \
  --run-name r3_agg64_s2026 --output-dir runs/route3_diag --epochs 120
.venv/bin/python scripts/train.py --config configs/route_delay_fno_r3_m32_s2027.json \
  --run-name r3_m32_s2027 --output-dir runs/route3_diag --epochs 120
```

训练默认：AdamW lr 3e-4 / wd 1e-4、cosine + 250 warmup steps、batch 64、grad clip 1.0、bf16、early stopping patience 20（val MAE）、max 120 epochs；每 5 epoch 在固定 200 个 validation states 上做官方 one-shot 评估（epoch 10 起）。

### 5.2 评估

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/evaluate.py \
  --config configs/route_delay_fno_r3_m32.json \
  --checkpoint runs/route3_modes/r3_m32_s2026/best_mae.pt \
  --split validation --limit 1000 --identification-limit 1000 \
  --batch-size 16 --output artifacts/analysis/oneshot_val_m32.json
```

### 5.3 导出（TorchScript）

```bash
.venv/bin/python scripts/export_model.py \
  --config configs/route_delay_fno_r3_m32.json \
  --checkpoint runs/route3_modes/r3_m32_s2026/best_mae.pt \
  --output-dir runs/route3_export --check-seed 777777
```

输出：`s_to_delta_m.pt` + `metadata.json`；CPU 加载，B=1/B=4 与 eager 数值一致（diff = 0.0）。

### 5.4 残差与失败分析

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/analyze_model.py \
  --config configs/route_delay_fno_r3_m32.json \
  --checkpoint runs/route3_modes/r3_m32_s2026/best_mae.pt \
  --split validation --limit 1000 --identification-limit 1000 \
  --output-root artifacts/analysis/final_m32 --tag validation
```

`analyze_model.py` 保存 residual npz、per-param MAE/P95、最难参数、失败 enrichment、worst states 与三张诊断图。


---

## 6. 实验结果

### 6.1 Stage 1 工程准备（摘要）

- 环境：Python 3.10.12 / torch 2.4.1+cu121 / RTX 4060 8 GB。
- 数据：8000 / 1000 / 1000，308 MB，sha256 校验通过，split 无重叠，public seed 零重叠。
- anchor：合规，min margin +0.6387 dB，40,322 calls / 227.1 s。
- 四条路线均可 forward/backward、TorchScript 导出；参数 2.31M–5.63M；tiny-overfit loss 降至初值 21%–32%。
- 22 个单元测试通过。

### 6.2 Representation study（R1/R2/R3，m64，同一 schedule）

| Representation | MAE | P95 | P99 | mean miss | median | P90 | worst | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 raw only | 0.03297 | 0.08454 | 0.11225 | 21.80 dB | 19.15 | 33.21 | 65.57 | 0 |
| R2 Tukey only | 0.03311 | 0.08464 | 0.11387 | 21.91 dB | 19.14 | 34.29 | 59.79 | 0 |
| **R3 raw + Tukey** | **0.03249** | **0.08421** | **0.11218** | 22.06 dB | 19.13 | 33.90 | **48.91** | 0 |

三条路线差异 ~1.5%，在 seed 噪声内；R3 略优且信息最完整，选 R3。Tukey α = 0.25。

### 6.3 FNO mode study（R3，其余完全一致）

| modes | params | val MAE | P95 | mean miss | worst | epochs | train |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **32** | **3.28M** | **0.03222** | **0.08330** | **21.05 dB** | **45.86** | 41 | 718 s |
| 64 | 5.63M | 0.03249 | 0.08421 | 21.45 | 47.31 | 33 | 588 s |
| 96 | 7.99M | 0.03275 | 0.08468 | 21.17 | 52.33 | 33 | 601 s |
| 128 | 10.35M | 0.03273 | 0.08428 | 21.30 | 46.03 | 31 | 581 s |

**结论：32 modes 足够且最优；64 不必作为起点，96/128 更大但更差。**

### 6.4 最终 m32 模型（validation 1000 / internal-test 1000）

| Metric | Train | Validation | Internal test |
|---|---:|---:|---:|
| MAE | 0.02739 | 0.03222 | 0.03247 |
| RMSE | 0.03567 | 0.04153 | 0.04185 |
| Median abs | — | 0.02607 | 0.02608 |
| P90 | — | 0.06923 | 0.06960 |
| P95 | 0.07243 | 0.08328 | 0.08403 |
| P99 | 0.09973 | 0.11155 | 0.11282 |
| Max | 0.19271 | 0.20005 | 0.22837 |
| ‖e‖₂ mean | 0.2262 | 0.2634 | 0.2655 |
| ‖e‖∞ mean | 0.0936 | 0.1056 | 0.1063 |

One-shot：

```text
validation   1000 states, pass 0, rate 0.000
             mean 20.95 dB, median 18.94, P90 29.62, worst 54.31
internal     1000 states, pass 0, rate 0.000
             mean 21.16 dB, median 18.97, P90 31.98, worst 59.93
```

- correction L2 mean 0.313，max 0.403；clipping 2.18%。
- 逐项通过率（validation）：S11_1 0%，S11_2 0%，S21_1 9.3%，S21_2 0.2%，S21_3 9.2%，S21_4 0.1%，S21_5 7.7%。

### 6.5 诊断结果

| 诊断 | 结果 |
|---|---|
| 均值基线 | MAE 0.03923 |
| kNN k=1 / 5 / 20 | 0.0481 / 0.0371 / 0.0352 |
| Ridge on PCA-256 | **0.02887**（优于 FNO ~10%） |
| Ridge PCA-1024 | 0.03016 |
| 64-sample 强过拟合 | train loss 0.00012（MAE ≈0.003），容量充足 |
| train/val gap | train MAE 0.02739 vs val 0.03222；后期 train loss↓、val↑ |
| agg-length 64 探针 | MAE 0.03157，mean miss 20.57，8.39M 参数，仍 pass=0（~2% 改善） |
| m32 第二 seed | val MAE 0.03204（seed 方差 ~0.7%） |

**可辨识性**：

- 平均 per-parameter R² = 0.2725，中位数 0.2293。
- 好：`0 (R²=0.963, MAE=0.0067)`、`3 (0.907, 0.0117)`、`2 (0.905, 0.0112)`。
- 差（R² < 0.1）：`1, 21, 23, 26, 29, 31, 33, 34, 37, 38, 39, 40`。
- 最难 MAE：`1 0.0409, 33 0.0407, 38 0.0402, 26 0.0399, 40 0.0399, 21 0.0397, 31 0.0394, 29 0.0388, 23 0.0380, 34 0.0379`。
- 单参数敏感度（δ=0.01）中位数 0.0229；最弱集合 `1, 40, 38, 39, 33, 35, 36, 31, 37, 34`，与低 R² 集合高度重合。
- 局部 Jacobian 条件数：anchor 238.9，seed0 509.5；最小相对奇异值 0.037 / 0.015。

**Anchor 容差（单参数扰动，实测）**：

| δ | 合规率 | 最坏 min margin |
|---:|---:|---:|
| 0.001 | 100% | +0.383 dB |
| 0.002 | 98.8% | −0.168 dB |
| 0.005 | 57.3% | −1.687 dB |
| 0.010 | 20.7% | −4.858 dB |
| 0.020 | 7.3% | −10.966 dB |

**失败机制**：

- 84% 的 state 由 S11 频段限制（48% / 36%），平均 miss ≈18.7 dB；S21 占 16%，但 miss 30–35 dB。
- `corr(‖e‖∞, max_miss)=0.0795`；按危险参数梯度加权的误差相关性 0.211 → 方向比幅值更重要。
- 用 anchor margin 梯度线性化预测“全部通过”，实际 miss ≈21 dB → 0.03 误差已使通带严重失配，属强非线性。
- 结论：要 one-shot 合规，单参数精度需 ~0.001–0.002；当前 MAE 0.032 约差 20–30 倍。

### 6.6 Q1–Q6 结论

1. **Best representation**：R3 raw+Tukey（α=0.25），但优势在噪声内。
2. **State accuracy**：MAE 0.0322，P95 0.0833，P99 0.112，‖e‖∞ mean 0.106。
3. **One-shot pass rate**：0/1000（validation 与 internal test 均为 0）。
4. **Failure pattern**：少数危险方向 + 大量弱可辨识方向；S11 匹配坍塌主导。
5. **64 modes 是否足够**：不需要 64；**32 已足够且最优**。
6. **限制类型**：**generalization/data-limited + 逆问题病态**；不是 capacity 或优化崩溃。


---

## 7. 已知问题

1. **one-shot pass = 0**：即使最好的 m32 模型，state 误差仍比容差大 20–30 倍；S11 带内几乎全反射。
2. **弱可辨识参数**：12 个参数 R²<0.1；这些方向在任何纯监督方案下都很难靠 8k 样本恢复。
3. **anchor 余量小**：+0.64 dB，单参数 0.005 扰动即有 42.7% 概率破坏合规；但即使扩大 anchor 余量也无法吸收 0.03 的误差。
4. **FNO/aggregation 不是主瓶颈**：agg64 只提升 ~2%；Ridge 反超 FNO ~10%，说明架构仍有优化空间，但不足以跨越数量级差距。
5. **fp16 AMP 不可用**：奇数长度 cuFFT half 精度不支持，部分路线溢出；已固定 bf16 + FFT float32。
6. **多进程 BLAS 线程超订**：数据/anchor 脚本必须设 `OPENBLAS/OMP/MKL_NUM_THREADS=1`，否则会卡住（本阶段曾因此重启一次 R2）。
7. TorchScript trace 对静态长度有 TracerWarning（2001/2129 固定），batch 维仍动态，已验证 B=1/4。

---

## 8. 建议下一步（未执行）

1. **扩大训练数据（最高优先级）**：8k → 50k–200k states。当前吞吐 ~190 states/s，50k 约 5 分钟；这是最可能把 MAE 降到 0.01 以下的杠杆。
2. **margin/sensitivity-aware loss**：对危险方向（`8, 32, 37, 35, 22, 23, 10, 24, 31, 5`）加权；对不可辨识方向降权；或直接用可微 surrogate 优化 correction 后的 7 项 margin。
3. **robust anchor refinement**：用真实 residual 分布优化 `m*_robust`，而不是继续最大化 nominal margin；但 anchor 只能辅助，不能替代 state 精度。
4. **保留 R3 + m32**；暂不扩大 channels/blocks；若允许改 readout，aggregation length 64 是低风险小改进。
5. **若目标是一次性高通过率**：考虑 unrolled/内部迭代 refinement 或 simulator-in-the-loop 可微 surrogate，把有效精度推到 1e-3。
6. 不建议现在进入 Route 1/2/4；先解决数据规模与目标函数。

---

## 9. 合规与注意事项

- **public seeds `10000–10049` 从未使用**；evaluator 有显式 guard（在调用 simulator 前抛出）。
- **anchor 未在 Stage 2 重新优化**，仍为 `+0.6387 dB`。
- **未修改 simulator**；仅使用公开 API。
- **未训练 Route 1/2/4 的正式模型**；R1/R2 仅作为 mandated representation study。
- internal test 仅在模型/representation/mode 选型完成后使用。
- 本仓库**不包含**模型权重、anchor 向量、norm stats、residual npz、cache、runs；这些为本地生成产物。若仓库为 public，请注意作业学术诚信条款（不得交换 generated targets / compliant vectors / model weights）。
- `assignments/` 为课程提供材料，除运行外不要修改。

---

## 10. 交接清单

先读：

1. 本文件 §3（contract）与 §6（结论）；
2. `solution/models.py`、`solution/trainer.py`、`solution/evaluator.py`、`solution/analysis.py`；
3. `scripts/train.py`、`scripts/analyze_model.py`。

本地关键文件（不在 git）：

```text
cache/dataset/                                    数据缓存 + manifest
artifacts/common/anchor.npz                       合规 anchor 向量
artifacts/common/norm_stats.npz                   归一化统计
artifacts/analysis/final_m32/validation/          最终模型 residual + summary + plots
artifacts/analysis/final_m32/internal_test/       同上
runs/route3_modes/r3_m32_s2026/best_mae.pt        最终参考 checkpoint
runs/route3_export/s_to_delta_m.pt                最终模型 TorchScript
runs/route3_repr/                                 R1/R2/R3 runs
runs/route3_modes/                                mode study runs
runs/route3_diag/                                 agg64 + seed2027 runs
```

建议的交接验证：

```bash
.venv/bin/python -m unittest discover -s tests -v          # 期望 22 tests OK
.venv/bin/python scripts/smoke_test.py --tiny-steps 30     # 期望 checks 全 true
.venv/bin/python scripts/evaluate.py --config configs/route_delay_fno_r3_m32.json \
  --checkpoint runs/route3_modes/r3_m32_s2026/best_mae.pt \
  --split validation --limit 200 --identification-limit 200
```

---

## 11. 命令速查

| 目的 | 命令 |
|---|---|
| 生成数据 + stats | `scripts/build_dataset.py --workers 12 --norm-configs ...` |
| 找 anchor | `scripts/find_anchor.py --workers 12 --population 256 --iterations 150 ...` |
| 训练 | `scripts/train.py --config <cfg> --run-name <name> --output-dir <dir> --epochs 120` |
| 评估 | `scripts/evaluate.py --config <cfg> --checkpoint <pt> --split validation --limit 1000` |
| 导出 | `scripts/export_model.py --config <cfg> --checkpoint <pt> --output-dir <dir>` |
| 残差分析 | `scripts/analyze_model.py --config <cfg> --checkpoint <pt> --split validation --limit 1000` |
| 单测 | `.venv/bin/python -m unittest discover -s tests -v` |
| smoke | `.venv/bin/python scripts/smoke_test.py --tiny-steps 30` |

> 多进程 / simulator 密集命令前加：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`


---

## 12. Stage 3 — Pairwise / observable-subspace control

### 12.1 动机与设计

Stage 2 表明 full 41-D state identification 的误差（MAE 0.032）远大于 one-shot 容差。Stage 3 检验另一种分解：

```text
(S_source, S_target) -> delta_m
Delta z = V_K^T (S_j - S_i)      (response PCA latent difference)
Delta u = u_j - u_i              (bounds-normalized parameter difference)
Delta z ~= A Delta u             (local Jacobian)
A = U S V^T                      (observable parameter subspace V_r)
c = V_r^T Delta u ~= W Delta z   (linear differential controller)
Delta m = half_range * V_r c
```

- 不生成新 state；只用缓存 train/validation/internal + 现有 anchor 的 response。
- PCA、A、SVD、V_r、W、ridge λ 只用 train split 拟合；λ 用 train 内部 holdout（states 7000–7999）选择。
- validation 只用于选择 r、controller 与 one-shot 评估；internal test 只在配置冻结后评估一次。
- anchor 未重新优化；public seeds 未使用。

### 12.2 实现与产物

```text
solution/pairwise.py          PCA / pair sampler / ridge A / ridge controller /
                              correction evaluation
scripts/pairwise_study.py     stages: pca | pairs | jacobian | residual | controllers | all

artifacts/analysis/pairwise_control/
  pca_spectrum.json                 explained / cumulative variance
  pair_indices.npz                  (i, j, type) for train/val (git-ignored)
  pair_distance_analysis.json       u-distance vs z-distance by pair type
  local_control_map_A.npz           A_K, U_K, S_K, Vh_K, lambda (git-ignored)
  singular_spectrum.json            spectrum / cumulative energy / condition
  observable_bases_r*.npz           V_r (git-ignored)
  residual_subspace_analysis.json   Stage-2 residual -> observable/weak
  linear_controller_results.json    pair-level metrics for C1/C2/C3 x K x r x λ
  validation_results.json           anchor one-shot grid + full validation + internal
  pairwise_summary.json             汇总
  ambiguity_direction_analysis.json response-similar / parameter-far directions
  supplementary_K_eval.json         K=128/256/512 anchor correction check
  anchor_pair_linear.json           直接 train->anchor 线性 controller 对照
  plots/                            singular spectrum / miss-vs-r / pass-vs-r
```

复现：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/pairwise_study.py --stage all --rebuild
```

### 12.3 结果

**Q1 — response PCA / observable spectrum**

```text
PCA cumulative explained: K=64 95.65%, 128 99.17%, 256 99.91%, 512 99.995%
A singular values (K=512): 21.22, 16.30, 13.38, 11.90, 10.06, ..., 0.20
condition number: 105.6 (K=512), 111.0 (K=128)
cumulative squared energy at r: 1 0.25, 3 0.50, 9 0.80, 13 0.91,
                                16 0.95, 22 0.99, 24 0.995, 30 0.999
holdout Delta-z normalized RMSE (local A): 0.785 (R^2 ~= 0.38)
```

**无sharp elbow**；按能量可用 r≈16（95%）或 r≈24（99.5%），但弱方向并非严格 null（σ_min/σ_max≈0.01），条件数约 106。

**Q2 — response 相近但参数差异大**

```text
train pairs: 351,988（local 128k, response-local 128k, medium 32k, random 64k）
u-distance mean: local 2.46, response-local 3.13, medium 3.82, random 3.61
corr(u-distance, z-distance): response-local 0.13, local 0.23, random 0.37
```

response 最近的 10% pair，其 parameter distance 仍有 3.11（random 3.59）：response 相似只带来约 13% 的参数距离缩减。歧义 pair 的 Δu 主要参数（mean |Δu| 排序）：`1, 34, 37, 28, 35, 27, 25, 30, 24, 32, 18, 20`；与前阶段弱可辨识集合部分重合。

**Q3 — 旧 Route-3 residual 在 observable/weak 子空间中的能量**

| r | observable fraction | weak fraction | corr(miss, obs norm) | corr(miss, weak norm) |
|---:|---:|---:|---:|---:|
| 8 | 0.097 | 0.903 | 0.129 | 0.224 |
| 12 | 0.162 | 0.838 | 0.147 | 0.216 |
| 16 | 0.242 | 0.758 | 0.160 | 0.204 |
| 20 | 0.334 | 0.666 | 0.186 | 0.188 |
| 24 | 0.437 | 0.563 | 0.209 | 0.162 |
| 32 | 0.688 | 0.312 | 0.228 | 0.116 |

在 r=8/16 时大部分 old error 位于“weak”子空间，但 weak 误差仍与 miss 相关（top-quartile weak 23.1–23.3 dB vs obs 21.9–22.2 dB）。说明弱方向在 0.03 误差尺度上并非真的无关（局部线性化失效）。

**Q4 — linear pairwise controller（K=512，300 validation states）**

| r | C1 local | C2 local+resp | C3 mixed |
|---:|---:|---:|---:|
| 8 | 33.15 | 33.27 | 33.33 |
| 12 | 32.41 | 32.34 | 32.39 |
| 16 | 32.16 | 32.07 | 32.01 |
| 20 | 24.78 | 23.55 | 23.40 |
| 24 | 21.99 | 21.06 | 20.88 |
| 32 | 21.97 | 20.98 | **20.46** |
| 41 | 21.97 | 21.08 | 20.68 |

（数值 = mean max specification miss，dB；所有配置 pass rate = 0）

- 最佳：**C3 mixed, r=32, K=512, λ=0.01**。
- Full validation 1000：pass **0/1000**，mean **20.60 dB**，median 18.70，P90 30.09，worst 71.74。
- Internal test 1000：pass **0/1000**，mean **20.26 dB**，median 18.73，P90 28.38，worst 51.22。
- 对照：旧 Route-3 state model（full validation）mean 20.95，median 18.94，P90 29.62，worst 54.31，pass 0。
- 官方 sanity：oracle anchor correction 300/300 pass（mean 0）；zero correction 0/300，mean 36.03。
- K 影响（C3 r=32，300 states）：K=128 20.49，K=256 20.52，K=512 20.46 → 几乎无影响。
- pair 级指标（C3 K=512）：r=32 的 val observable-coordinate normalized RMSE 0.752，Δz 重建 R² 0.513；r=41 为 0.821 / 0.513。
- 直接 train→anchor 线性对照（anchor_pair_linear）：最佳 r=41，300-state mean 21.89，full validation 22.09，internal 21.70，pass 0。

**Q5 — 是否出现非零 pass rate**

没有。所有 r、C1/C2/C3、K=128/256/512、以及 train→anchor 直接线性拟合，pass rate 均为 **0%**。

**Q6 — 是否值得训练 neural pairwise controller**

**No。** 按预设停止条件：

- mean miss 最好 20.46/20.60 dB，相对旧 20.95 dB 仅改善约 0.35–0.5 dB（且 P90/worst 变差）；
- 非零 pass rate 未出现；
- Δz 重建 R²≈0.51，不属于“很好”；
- 直接 anchor-pair 线性拟合更差（21.9–22.1 dB）。

因此不训练 neural controller，Stage 3 到此停止。

### 12.4 结论

1. **response latent 的有效维度**：无 sharp elbow；按能量 r≈16–24，但弱方向仍非零（条件数 ~106），不存在干净的低维可观测子空间。
2. **response 相似 ≠ 参数相似**：最近 response 的 pair 参数距离仍接近随机，说明 full-state inversion 的 weak-observability/null-space 问题真实存在。
3. **线性差分控制不成立**：从 Δz 线性预测 Δu/observable coordinates，在 anchor 大跨度 correction 上没有产生可用信号；mean miss 与旧 state-identification 基本相同，pass 0。
4. 原因判断：anchor 步长较大、局部线性 A 只解释 ~38% 的 Δz 方差、参数-响应映射强非线性且病态；线性化不足以支撑 0.001–0.002 的 correction 精度。
5. 这并不否定“直接预测 correction”的方向，但说明**在当前 anchor、PCA latent、线性控制器与 8k 数据下**不可行。

---

## 13. Stage 4 — Per-state compliant teacher pilot

### 13.1 动机

检验假设：旧的固定 anchor target `m* - m_i` 是否把大量“只在弱可辨识方向上的移动”强加给了网络；若为每个 state 找一个**离它更近的合规点**，监督标签是否会显著更 sparse / low-rank / 可学习。

约束：不生成新 state；只对 train 前 256 个 state 做 pilot；anchor 固定；未使用 validation 调 teacher optimizer；public seeds 未使用。

### 13.2 实现

```text
solution/teacher.py        line_search_anchor / local_teacher_search / margins
scripts/teacher_pilot.py   line | teacher | analyze | all

artifacts/teacher/pilot/
  line_gamma0.00.npz/json       current -> anchor 线搜索
  line_gamma0.25.npz/json       robust margin = 0.25 dB
  teacher_gamma0.00.npz/json    per-state 局部可行点搜索
  teacher_gamma0.25.npz/json
  analysis_gamma0.00.json       几何 / sparsity / SVD / projection / 一致性 / Ridge
  analysis_gamma0.25.json
  mean_baseline.json            target 均值预测基线
  plots/                        t_line / reduction / teacher spectrum / sparsity
```

复现：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/teacher_pilot.py --stage all \
  --pilot 256 --workers 12 --gammas 0.0 0.25 --max-iter 8 --population 32
```

局部搜索：从 line teacher 出发，方向 = 指向 current 的 `d0` + Stage 3 observable/weak 方向 +
9 个 specification-sensitive 参数 one-hot + random；每轮 32 候选；lexicographic 目标：
(1) `min margin >= gamma`，(2) 最小 `||Delta u||_2`，(3) 平局时更大 min margin。

### 13.3 结果

**Q1 — anchor 连线边界**

| gamma | t median | P10 | P25 | P75 | P90 | min | max | t<0.25 | t<0.50 | t<0.75 | t>0.90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | **0.9867** | 0.9777 | 0.9828 | 0.9898 | 0.9922 | 0.9609 | 0.9953 | 0 | 0 | 0 | 100% |
| 0.25 | 0.9898 | 0.9824 | 0.9859 | 0.9922 | 0.9938 | 0.9633 | 0.9969 | 0 | 0 | 0 | 100% |

沿 current→anchor 连线，必须走到几乎 anchor 本身才合规；不存在“更早进入合规区”的现象。

**Q2 — 局部可行点搜索**

| gamma | pass | teacher/anchor norm median | mean | P10 | P90 | min margin median |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 256/256 | **0.9696** | 0.9680 | 0.9539 | 0.9797 | 0.076 dB |
| 0.25 | 256/256 | 0.9724 | 0.9715 | 0.9574 | 0.9837 | 0.318 dB |

局部搜索只能比 anchor 近约 **3%**（line 本身为 0.987，teacher 0.970）。

**Q3 — sparsity / 低秩**

```text
|Delta u| > 0.01 : mean 40.25 / 41 parameters
|Delta u| > 0.05 : mean 37.10
|Delta u| > 0.10 : mean 33.22
teacher correction singular values: 29.58, 9.45, 9.03, 8.95, 8.56, ...
cumulative variance: r=1 0.364, r=4 0.468, r=12 0.688, ...
```

不 sparse；看似“低秩”主要是因为所有 correction 共享同一个 anchor 方向，而不是个体只动少数参数。

**Q4 — observable / weak 子空间能量**

| r | teacher observable fraction | anchor observable fraction |
|---:|---:|---:|
| 8 | 0.1355 | 0.1353 |
| 16 | 0.4564 | 0.4586 |
| 24 | 0.7023 | 0.7030 |
| 32 | 0.8415 | 0.8427 |

teacher 与 anchor 几乎完全相同：**没有减少弱子空间上的移动**。

**Q5 — response-neighbor 一致性**

```text
nearest-response teacher delta distance : 3.135
random teacher delta distance          : 3.474   (ratio 0.903)
nearest-response anchor delta distance : 3.227
random anchor delta distance           : 3.579   (ratio 0.902)
corr(response distance, teacher delta) : 0.339
corr(response distance, anchor delta)  : 0.350
```

teacher 标签相对 anchor 没有变得更 response-consistent。

**Q6 — cheap learnability（pilot 内 80/20 split，Ridge，带截距）**

| target | K=64 nrmse | K=512 nrmse | mean-predictor baseline |
|---|---:|---:|---:|
| teacher Δu | 0.740 | 0.805 | 0.822 |
| anchor Δu | 0.738 | 0.803 | 0.820 |
| absolute state u | 0.902 | 0.982 | 1.005 |

teacher target 与 anchor target 几乎一样；相对 absolute-state 的“提升”主要来自 target 共有的均值方向（mean baseline 也相应更低），
没有可辨别的 “teacher 更可学” 信号。

**Q7 — 是否扩展到 8000**

**No。** 五项预设成功指标全部未满足：

```text
teacher norm 显著降低？      3% only
effective rank 明显降低？    否（低秩仅来自共享均值方向）
response-neighbor 更一致？   否（ratio 0.903 vs anchor 0.902）
weak-subspace 能量降低？     否（projection 几乎相同）
Ridge 明显更好学？           否（teacher ≈ anchor）
```

Simulator calls：line 2×2,048 + teacher 93,038 + 91,796 = **188,930**；总 wall ~1,039 s。

### 13.4 结论

当前 anchor 附近的合规区是一个很小的“口袋”：256 个 train states 沿任意测试方向都只能把 correction 缩短约 3%。
per-state teacher 与 fixed-anchor correction 在 norm、方向、sparsity、observable/weak 分布和 response 一致性上几乎不可区分。

因此：**“旧标签是错的、真正需要的是 per-state 更近的合规目标”这一假设在本 anchor 下不成立。**
问题不在监督标签的选择，而在于 response 对合规 correction 的信息不足/条件数差。

不建议对 8000 states 生成 teacher；若未来重开此方向，需要先改变 compliance target 族（多 anchor / 更宽 margin）
或大幅扩充数据，而不是继续优化 teacher。


---

## 14. Stage 5 — System-identification / rational representation

### 14.1 目标与约束

把 `S11,S21` 从“4×2001 real sequence”重新视为有限阶 LTI 复频率响应，测试 rational /
system-identification 特征是否让 41-D 参数（或 anchor correction）更易恢复。

- **不扩充 state 数据**：只用缓存 train 512 → 2000、validation 1000；未调用 `sample_state`。
- 不读取 simulator internals；只用公开频率网格与复响应。
- 未使用 public seeds；未修改 simulator；未重新优化 anchor。

### 14.2 实现

```text
solution/system_id.py          Chebyshev shared-denominator rational fit、
                              frequency-holdout 重建、spec 重建、系数/极点零点、
                              complex algebra / log-derivative features
scripts/system_id_pilot.py     sweep | rational | algebra | regress

artifacts/system_id/
  sweep_512.json                     order sweep（512 states，75%/25% 频率 holdout）
  rational_train.npz / .json         2000 states 的 coefficients / features
  rational_validation.npz / .json    1000 states
  algebra_regression_2000.json       F1/F2/F3/log-deriv 的 PCA-Ridge
  rational_regression_2000.json      R1/R2/R3 的 PCA-Ridge
  oneshot_r3_order64.json            best cheap model 的官方 one-shot
  mlp_rational_features.json         small MLP 对照
  stage5_summary.json                汇总
```

Rational 模型（shared denominator）：

```text
D(x)  = 1 + sum_{k=1..p} d_k T_k(x)
N11(x)= sum a_k T_k(x),  N21(x)= sum b_k T_k(x)
S11=N11/D, S21=N21/D
x in [-1,1]（频率线性映射），complex least squares + Tikhonov，列归一化
fit: 75% frequency bins, evaluate: 25% held-out bins
```

复现：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/system_id_pilot.py --stage sweep --n-sweep 512 --workers 12
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/system_id_pilot.py --stage rational --orders 48 64 --n-rational 2000 --workers 6
.venv/bin/python scripts/system_id_pilot.py --stage algebra --n-algebra 2000
.venv/bin/python scripts/system_id_pilot.py --stage regress --orders 48 64 --n-rational 2000
```

### 14.3 Rational fit（Q1/Q2）

512 个 train states，λ=1e-10，75% 频率拟合 / 25% holdout：

| order | holdout relL2 median | relL2 p90 | spec err median (dB) | spec err p90 | rel>1e-3 states | nonfinite | cond median |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.0321 | 0.0753 | 8.28 | 19.8 | 512/512 | 0 | 6.3e8 |
| 40 | 0.00942 | 0.0249 | 2.59 | 12.8 | 502/512 | 0 | 8.7e8 |
| 48 | 3.92e-4 | 2.16e-3 | 0.132 | 2.38 | 135/512 | 0 | 1.2e9 |
| 56 | 1.06e-5 | 7.93e-5 | 0.0034 | 0.074 | 1/512 | 0 | 1.9e9 |
| 64 | 1.41e-6 | 9.99e-6 | 0.00047 | 0.013 | 0/512 | 0 | 3.3e9 |

**Q1：可以。** order 56/64 在未参与拟合的频率 bins 上 complex relL2 ~1e-5–1e-6，七项 spec summary 误差 ~1e-3–1e-2 dB；512 个 state 均无 nonfinite。
**Q2：有效 order ≈ 48–64。** order 48 已可用（relL2 4e-4，spec 0.13 dB），order 64 近乎精确。

### 14.4 Complex algebra features（Q5）

2000 train / 1000 validation，PCA + Ridge（λ 用 train 内部 split 选择），target = absolute u：

| feature | best physical MAE | mean R² | #R²<0.1 |
|---|---:|---:|---:|
| raw response | **0.02950** | 0.342 | 15 |
| 1/S21 (eps 1e-6) | 0.03765 | 0.066 | 30 |
| S11/S21 (eps 1e-6) | 0.03768 | 0.065 | 31 |
| combined (eps 1e-6) | 0.03722 | 0.082 | 27 |
| log-derivative (eps 1e-6) | 0.03762 | 0.072 | 29 |

**Q5：没有额外信息。** 所有 complex algebra / log-derivative 变换都明显差于 raw response。

### 14.5 Rational features（Q3/Q4）

| feature | best physical MAE | mean R² | #R²<0.1 | K |
|---|---:|---:|---:|---:|
| raw response | 0.02950 | 0.342 | 15 | 64 |
| R1 order48 coefficients | 0.02622 | 0.456 | 4 | 256 |
| R1 order64 coefficients | 0.02601 | 0.464 | 4 | 256 |
| R3 order48 + raw PCA64 | 0.02503 | 0.501 | 4 | 256 |
| R3 order64 + raw PCA64 | **0.02502** | **0.504** | **3** | 256 |
| R2 poles/zeros order48 | 0.03375 | 0.234 | 10 | 256 |
| R2 poles/zeros order64 | 0.03482 | 0.178 | 18 | 256 |

**Q3：系数特征明显更好；极点/零点不稳定。** R1/R3 相对 raw 的 MAE 降低约 **15%**，mean R² 0.34→0.50，R²<0.1 参数 15→3；R2（排序后的 roots）反而比 raw 更差，说明 root/permutation conditioning 不可靠，不采用。

**Q4：weak parameters 显著改善（raw → R1 order64 / R3 order64）**

| param | raw R² | R1-64 R² | R3-64 R² |
|---:|---:|---:|---:|
| 1 | −0.026 | 0.617 | 0.623 |
| 37 | 0.009 | 0.561 | 0.572 |
| 38 | 0.009 | 0.385 | 0.372 |
| 39 | 0.069 | 0.667 | 0.696 |
| 40 | 0.014 | 0.260 | 0.309 |
| 34 | 0.128 | 0.485 | 0.487 |
| 29 | 0.070 | 0.209 | 0.199 |
| 33 | 0.017 | 0.157 | 0.162 |
| 21 | 0.085 | 0.077 | 0.086 |
| 23 | 0.033 | 0.010 | 0.023 |
| 26 | 0.064 | 0.074 | 0.115 |
| 31 | 0.021 | 0.067 | 0.074 |

Dangerous parameters 同样改善：32 `0.177→0.469`，35 `0.075→0.594`，37 `0.009→0.572`；
8/5/10 基本保持（0.81–0.90）。

### 14.6 Small MLP 对照

452-dim R3 features → 256 → 256 → 41（192,297 参数）：

```text
physical MAE 0.02731, mean R² 0.439, #R²<0.1 = 6
```

**不如 Ridge**（0.02502 / 0.504 / 3）；在 2000 样本上 MLP 没有额外收益。

### 14.7 One-shot（Q6）

Best cheap model：Ridge on R3(order64)+raw64，K=256，validation 1000：

```text
pass 0/1000 (0%)
mean 20.50 dB, median 18.69, P90 30.76, worst 56.23
correction L2 mean 0.340, clipping 0.96%
```

对比旧 Route-3 state model：mean 20.95 dB，pass 0。  
**Q6：没有显著改善。** 表示层提升了 state 回归，但仍远不足以跨越 compliance 精度门槛。

### 14.8 Q7 — realization / parameterization ambiguity

Rational order 64 对 response 的重建几乎精确（holdout relL2 ~1e-6，spec 误差 <0.001 dB），
但仍有 4 个参数 R²<0.1：**21, 23, 26, 31**（另有 33/29 较低）。
这提供了明确的 **response-equivalent / near-equivalent hidden-state ambiguity** 证据：

> 频响可以被低阶 rational model 精确表示，但 41 维隐藏参数化中的部分方向仍不可从 S 唯一恢复。

因此不能把问题简单归结为“更多数据”；至少有一部分是 realization/parameterization ambiguity。

### 14.9 结论与 Stop

- Rational/system-ID 表示是**真实的正向信号**（weak parameters 大幅改善，MAE −15%）。
- 但 MAE 仍 0.025、one-shot mean 20.5 dB / pass 0，仍落在“未解决核心问题”的区间（0.025–0.03）。
- 按 §17/§21：**不扩展到 8000 states、不训练 FNO/大模型**。本阶段停止。
- 若继续，优先方向应是：在 rational coefficients 上结合 margin/sensitivity-aware 目标、
  或研究 canonical realization/control map，而不是增加样本或换更大网络。


