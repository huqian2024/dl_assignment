# CM19 One-Shot S→ΔM — 工程交接文档

> 最后更新：2026-11-18
> 项目根目录：`/home/qian/DL_assignment_2026`  
> 上游仓库：<https://github.com/huqian2024/dl_assignment.git>

---

## 立即恢复（新对话先读这里）

**当前状态：Stage 10 已完成并封存，停在 S11-head negative gate。不要继续训练或解冻。**

- 当前代码基线：`main`，Stage 10 功能提交 `63ffb59`；封存提交见 `git log -1`。
- 已完成：Stage 1–10。Stage 9 得到当前最佳 one-shot policy：
  validation mean **14.66 dB**（21.56→14.66），P90 **16.74 dB**，但 pass 仍为 **0/1000**。
- Stage 10 结论：
  - Stage 9 policy 高度依赖 clipping：**46.8%** coordinates 越界，**100%** 样本至少一个坐标被 clip。
  - S11 gradient 不弱（median norm **9.63** > S21 **4.53**），与 S21 基本协同（cosine **0.872**）。
  - S11 action space 很低维：**d=9** 解释 **95.7%**。
  - exact oracle 能把 S11 从 12.62→**8.10 dB**、overall→**9.34 dB**，S21→0.025；有真实下降路径。
  - 但 S11-specific head 只把 S11 从 12.87 降到 **12.69 dB**、overall 14.15→13.98，**pass 0**；head 失败。
- 因此**没有**跑 full S11 head validation，**没有** partial encoder unfreeze。
- 下一步唯一优先级：诊断 **S11 head 动作幅度/优化问题**（`c` mean |c| 仅 0.062），
  仍只在 train-tune 7000:7999 上试；head gate 通过后才允许 full-8000 + validation once。
- 硬约束：不新增 state/seed/public seed；不重优化 anchor；不训练 FNO/Transformer；
  不在 validation 上拟合；没有 head gate 不做 partial unfreeze。

详细证据：§18（Stage 9）、§19（Stage 10）、§20（本次封存与恢复清单）。

---

## 0. 当前状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| Stage 1 | 工程准备：环境、数据缓存、anchor、DSP、FNO/ResNet/Pooling/Head、四条路线、export、smoke test | ✅ 完成 |
| Stage 2 | Route 3 正式研究：representation R1/R2/R3、FNO modes 32/64/96/128、state/one-shot/residual 分析 | ✅ 完成 |
| Stage 3 | Pairwise / observable-subspace control 研究：response PCA、local Jacobian、可观测子空间、线性差分 controller | ✅ 完成（线性无信号，按规则停止） |
| Stage 4 | Per-state compliant teacher 结构验证（256-state train pilot，线搜索 + 局部可行点搜索 + 几何/可学习性分析） | ✅ 完成（无积极信号，不扩展到 8000） |
| Stage 5 | System-identification / rational representation（complex algebra、shared-denominator fit、Ridge/MLP） | ✅ 完成（表示层有正信号，但 one-shot 仍 0；按规则不扩到 8000） |
| Stage 6 | 现有 8000 状态 system-ID + ambiguity check（Vector Fitting、order-64、Ridge/ResMLP、学习曲线、最近邻） | ✅ 完成（最佳 physical MAE 0.02329，one-shot 仍 0） |
| Stage 7 | rational-coordinate stability / probes（稳定性、PCA、PLS2、ablation、quadratic） | ✅ 完成（系数稳定；未超过 Stage 6；不转 canonical/Loewner） |
| Stage 8 | learned-forward surrogate / refinement gate（raw-PCA、rational+raw、Adam/LM） | ✅ 完成（surrogate 不够准，refinement gate 未过；不 unroll） |
| Stage 9 | smooth specification + zero-order residual policy（calibration、SPSA、frozen-base head） | ✅ 完成（validation 21.56→**14.66**，pass 仍 0；S11 是瓶颈） |
| Stage 10 | S11-targeted gate（clipping、exact gradients、SVD、oracle、low-D SPSA、S11 head） | ✅ 完成；**S11 head negative gate，停止** |

**当前最佳 one-shot policy**：Stage 9 full-8000 refined policy
- validation mean **14.66 dB**，median 14.83，P90 16.74；
- pass **0/1000**；S21 已基本解决，两个 S11 requirements 仍 0% pass；
- Stage 10 的 exact oracle 证明 S11 有低维下降路径，但当前 S11 head 没有学会。

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

### 14.8 Q7 — 修正说明（Stage 6 复核）

Stage 5 曾把“rational 重建几乎精确 + 少数参数 Ridge R² 低”解释为 response-equivalent /
parameterization ambiguity 的证据。这个解释**证据不足**，Stage 6 已正式复核并撤回：

- rational 表示保留了 response 信息，只能说明 remaining inverse map 对线性 Ridge 仍困难；
- Stage 6 在 8000 条已有 train states 上做 exact raw-response nearest-neighbor：
  `d_S` 最小值 **0.258**、中位数 **0.390**，没有任何 `d_S < 0.01` 的 pair；
- 因此不能把低 R² 直接称为 ambiguity，也没有证明 control ambiguity。

Stage 6 的 near-collision / cross-correction 结果见 §15.6–15.7。

### 14.9 结论与 Stop

- Rational/system-ID 表示是**真实的正向信号**（weak parameters 大幅改善，MAE −15%）。
- 但 MAE 仍 0.025、one-shot mean 20.5 dB / pass 0，仍落在“未解决核心问题”的区间（0.025–0.03）。
- 按 Stage 5 stop gate：本阶段停止；Stage 6 已用**已有 8000 条数据**继续验证，
  没有新增 state，也没有重新优化 anchor。
- 若继续，优先方向应是：在 rational coefficients 上结合 margin/sensitivity-aware 目标、
  或研究 canonical realization/control map，而不是增加样本或换更大网络。



---

## 15. Stage 6 — shared-pole 试验、8000-state system-ID 与 ambiguity 复核

### 15.1 范围与硬约束

- 只使用已有缓存：train **8000**、validation **1000**；没有调用 `sample_state`，没有新增 seed。
- 未使用 public seeds `10000..10049`；未重新优化 anchor；未训练 FNO/Transformer。
- 将 Stage 5 的 order-64 Chebyshev rational coefficients 从 2000 条扩展到**全部已有 8000 条 train**；
  validation 1000 只用于评价。
- 代码：`solution/vector_fitting.py`、`scripts/stage6_study.py`；
  结果：`artifacts/system_id/vf_pilot.json`、`artifacts/stage6/*.json`。

### 15.2 Shared-pole Vector Fitting（Q1/Q2）

实现 normalized `s = j(ω-ω_c)/ω_s`、deterministic conjugate pole 初始化、shared poles、
5–6 次 relocation、稳定性/共轭约束、75/25 frequency holdout。32 个 train states 的 pilot：

| order | holdout relL2 median | P90 | spec max error median | pole movement median |
|---:|---:|---:|---:|---:|
| 16 | 0.3213 | 0.4774 | 28.22 dB | 8.8e-6 |
| 24 | 0.2876 | 0.3194 | 24.50 dB | 0.0646 |
| 32 | 0.2228 | 0.2922 | 19.83 dB | 0.2003 |
| 40 | 0.1788 | 0.2645 | 16.54 dB | 0.3416 |
| 48 | 0.1410 | 0.2199 | 14.16 dB | 0.4883 |

Fixed-pole barycentric/partial-fraction fallback：

| order | holdout relL2 median | P90 |
|---:|---:|---:|
| 64 | 0.0899 | 0.1181 |
| 96 | 0.0585 | 0.0845 |

对比 Stage 5 Chebyshev order 64：holdout relL2 median **1.19e-6**、spec error median **0.00039 dB**。
Vector Fitting 在所有测试 order 都远弱于 Chebyshev；pole movement 未收敛（order48 仍 0.49），
condition number 达 `1e12–1e13`，没有“最低可靠 order”。

**Q1/Q2 结论：** 本实现下 shared-pole VF 没有可靠 order，也不比 order64 Chebyshev 更稳定/更低维/更易预测；
按 §25 停止 VF 分支，保留 Stage 5 coefficient representation。

### 15.3 分组 whitening 与最终 feature

对 order64 系数按 `d / a / b` 三组做 train-only mean/std + PCA whitening：

```text
total 64  -> d/a/b = 21/22/21
total 128 -> 42/43/43
total 256 -> 84/86/86
total 388 -> 128/130/130 (full)
```

最终最好 feature 为 **order64 whitened 256-D + raw response PCA64**（320-D，记 V4）。
没有把 raw coefficients 直接送回归器。

由于 VF reconstruction 本身不可用（order48 仍 relL2 0.141），
V1（poles only）、V2（poles + residues）、V3（V2 + order64 latent）没有进入参数回归；
这是按 §25 停止 VF 分支的直接结果，而不是“pole 表示已被否定”。





### 15.4 8000-state models（Q3）

Validation 1000；`MAE` 为 bounds-normalized `u` 空间，`physical MAE` 为原始参数尺度。

| model | normalized MAE | physical MAE | P95 norm | mean R² | median R² | #R²<0.1 | weak 21/23/26/31 R² |
|---|---:|---:|---:|---:|---:|---:|---|
| Global Ridge (V4, 320-D) | **0.1780** | **0.02329** | 0.5184 | 0.564 | 0.562 | **1** | 0.153 / 0.083 / 0.190 / 0.144 |
| Local weighted Ridge (k=256, λ=10) | 0.2094 | 0.02758 | 0.6078 | 0.386 | 0.439 | 10 | −0.284 / −0.472 / −0.335 / −0.210 |
| Small ResMLP (1.36M) | 0.1770 | 0.02372 | **0.4933** | **0.572** | **0.610** | 4 | 0.038 / −0.061 / 0.031 / 0.023 |
| Stage 5 Ridge reference | — | 0.02502 | — | 0.504 | — | 3 | 0.086 / 0.023 / 0.115 / 0.074 |

- Global Ridge 的 physical MAE 从 Stage 5 的 0.02502 降到 **0.02329**（约 −7%），mean R² 0.504→0.564，
  `#R²<0.1` 从 3 降到 1。
- Local weighted Ridge 明显差于 global：在 whitened system-ID 空间中，近邻并不对应参数近邻；
  `k=32..256`、`λ=1e-2/10` 的最好结果仍只有 0.2094，因此没有“局部近似线性”的证据。
- ResMLP 的 normalized MAE/P95 略好，但 physical MAE 不如 Ridge，且 21/23/26/31 的 R² 反而更差；
  在这批 fixed-length system-ID 特征上没有看到非线性模型的实质增益。

### 15.5 Existing-data learning curve（Q4）

固定同一 validation 1000，只用已有 train 的前 N 条：

| N | Ridge MAE | Ridge mean R² | ResMLP MAE | ResMLP mean R² |
|---:|---:|---:|---:|---:|
| 1000 | 0.20571 | 0.4064 | 0.21998 | 0.4046 |
| 2000 | 0.18928 | 0.5040 | 0.20592 | 0.4653 |
| 4000 | 0.18217 | 0.5432 | 0.18857 | 0.5279 |
| 8000 | **0.17800** | **0.5643** | **0.17700** | **0.5721** |

曲线在 1k→8k 仍单调下降，但边际收益递减；8k 处 ResMLP 仅比 Ridge 低 0.001。
**只陈述现有曲线，不建议新增数据。**


### 15.6 True near-collision test（Q6）

在 8000 条 train 上做 exact raw-response nearest-neighbor（flatten complex S11/S21，relative L2）：

| statistic | value |
|---|---:|
| min `d_S` | **0.2580** |
| median nearest `d_S` | 0.3897 |
| nearest 1% median `d_S` | 0.2930 |
| nearest 0.1% median `d_S` | 0.2676 |
| nearest 0.01% median `d_S` | 0.2580 |
| pairs with `d_S < 0.01` | **0** |
| pairs with `d_S < 1e-3/1e-4/1e-5` | 0 / 0 / 0 |

最接近的 pair：seed `5914 ↔ 1729`，`d_S = 0.2580`，`d_u = 3.238`；
其后 `558 ↔ 3855`，`d_S = 0.2663`，`d_u = 3.103`。
`d_u` 的量级接近 normalized state 空间的直径，说明这些只是“相对接近”，**不是 response-equivalent**。

feature-space NN 更差：256-D whitened 特征最近邻 `d_S` 中位数 0.510、最小 0.285。
因此不能依赖 feature NN 作为 near-collision 候选。

### 15.7 Cross-correction 与 ambiguity 判据（Q7）

对 raw-response 最近的 30 个 pair 交叉应用 anchor correction（每对 4 种）：

| case | pass rate | 说明 |
|---|---:|---|
| `i + Δ_i`（own） | **100%** | anchor 本身合规 |
| `j + Δ_j`（own） | **100%** | 同上 |
| `j + Δ_i`（cross） | **0%** | miss 18–50 dB |
| `i + Δ_j`（cross） | **0%** | miss 18–50 dB |

但所有这些 pair 的 `d_S` 仍在 0.258–0.30，**不满足 `d_S << 1`**。
按 §20 的严格判据，cross correction 失败**不能**解释为 response ambiguity：它只反映了参数差异仍然很大。
因此：

> 现有 8000 条缓存 states 中**没有发现真正的 near-collision**；


### 15.8 Weak parameters 21/23/26/31 的条件 spread（Q5）

对每个 validation state 取 k=20 个 response-nearest train states，统计每个参数的条件 std：

| param | Ridge R² | ResMLP R² | raw-NN conditional/std | feature-NN conditional/std |
|---:|---:|---:|---:|---:|
| 21 | 0.153 | 0.038 | 0.936 | 0.879 |
| 23 | 0.083 | −0.061 | 0.927 | 0.914 |
| 26 | 0.190 | 0.031 | 0.944 | 0.916 |
| 31 | 0.144 | 0.023 | 0.947 | 0.908 |

同时 Stage 5 已改善的参数 1/37/38/39/40 的 raw-NN 条件 std 比值为 0.967/0.937/0.959/0.950/0.961，
与 21/23/26/31 没有本质差异。由于 raw NN 的 `d_S` 仍有 0.26–0.39，条件 spread 几乎不收缩是预期的；
**这不是 weak-observability 的证据**。更准确的结论是：

> 21/23/26/31 对当前 global nonlinear inverse map 仍然难预测；
> 但在现有数据的 response 分辨率下，没有证据表明它们不可辨识。

### 15.9 One-shot（Q8）

Validation 1000，best-anchor correction：

| model | pass | mean miss | median | P90 | worst |
|---|---:|---:|---:|---:|---:|
| Global Ridge V4 | 0/1000 | 19.80 dB | 18.49 | 26.17 | 54.93 |
| Local weighted Ridge | 0/1000 | 20.02 dB | 18.74 | 24.42 | 92.87 |
| **Small ResMLP** | **0/1000** | **18.96 dB** | **18.13** | **19.59** | **54.83** |
| Stage 5 Ridge reference | 0/1000 | 20.50 dB | 18.69 | 30.76 | 56.23 |

- ResMLP 平均 miss 比 Stage 5 低 **1.54 dB**，P90 从 30.76 降到 19.59；但仍 pass 0。
- Global Ridge 的 physical MAE 更低，但 one-shot mean miss 略高于 ResMLP；
  两者都远未达到 one-shot 所需的 `1e-3–2e-3` state precision。
- 没有出现 `pass > 0`，也没有出现 mean miss <15 dB；因此 Stage 6 只是**缩放信号**，不是问题解决。

### 15.10 Q1–Q8 汇总

1. **Q1**：shared-pole VF 在 order 16–48 都没有可靠 order；order48 仍 relL2 0.141 / spec 14.2 dB。
2. **Q2**：没有优于 order64 Chebyshev；VF 不稳定且精度差，已停止该分支。
3. **Q3**：8000 states 下 Global Ridge physical MAE **0.02329** / mean R² **0.564**；
   local weighted Ridge **0.02758** / 0.386；small ResMLP **0.02372** / 0.572。
4. **Q4**：Ridge 0.2057→0.1893→0.1822→0.1780；ResMLP 0.2200→0.2059→0.1886→0.1770；
   仍下降但边际递减；不建议新增数据。
5. **Q5**：21/23/26/31 仍是 global inverse map 的难点（Ridge R² 0.08–0.19，ResMLP 更差），
   但在 response-near 条件下没有观察到选择性的大 spread；不能判为不可辨识。
6. **Q6**：没有真正的 near-collision；raw-response 最小 `d_S = 0.258`，无 `d_S < 0.01`。
7. **Q7**：最近 pair 的 cross correction 全部失败，但 `d_S` 仍约 0.26–0.30，
   不满足严格 ambiguity 判据；不能据此声称 control ambiguity。
8. **Q8**：best one-shot 为 ResMLP：pass 0/1000，mean **18.96 dB**，median 18.13，
   P90 19.59，worst 54.83；比 Stage 5 的 20.50 dB 好，但仍远未通过。

### 15.11 Stop

- 已完成 shared-pole pilot、8000-state order64 特征、global/local/ResMLP、1k/2k/4k/8k 曲线、
  near-collision/cross-correction、validation one-shot。
- **未新增任何 state，未使用新 seed，未使用 public seeds，未重新优化 anchor，未训练大 FNO/Transformer。**
- 下一步如继续，只能在现有数据上做 margin/sensitivity-aware 回归或 canonical control map；
  不应把“更多数据”作为默认建议。

> Stage 5 的“ambiguity”解释不成立，Stage 6 也没有证明 control ambiguity。


---

## 16. Stage 7 — rational-coordinate stability、full-rank 与 probes

### 16.1 范围

- 只使用已有 train 8000 / validation 1000；未 `sample_state`、未新增 seed、未使用 public seeds、未重优化 anchor。
- 代码：`scripts/stage7_study.py`；结果：`artifacts/stage7/*.json`。
- 没有训练 FNO/Transformer；没有用 validation 拟合 residual。

### 16.2 Coefficient stability audit（Q1）

256 个 train states × 3 个 deterministic 75/25 frequency masks（A/B/C）× 3 个 λ（3e-11/1e-10/3e-10），
共 2304 次 order64 shared-denominator 拟合；denominator 固定为 `D=1+Σ d_k T_k`，没有 scale ambiguity。

| 指标 | median | P90 | P99 | max |
|---|---:|---:|---:|---:|
| coefficient relative L2 variation | **1.87e-4** | 2.74e-4 | 2.33e-3 | 2.85e-2 |
| coefficient cosine | ≈1.0 | ≈1.0 | ≈1.0 | ≈1.0 |
| probe relative variation | **8.52e-6** | 1.90e-4 | 1.41e-3 | 4.29e-3 |
| reconstruction relL2 | 1.30e-6 | 1.30e-5 | 6.12e-5 | 2.78e-4 |
| spec error (dB) | 3.84e-4 | 1.15e-2 | 2.45e-1 | 4.05 |
| fit condition number | 3.11e9 | 5.10e9 | 7.25e9 | 1.19e10 |

Per-group coefficient variation (median)：D **3.27e-4**，S11 numerator **1.43e-4**，S21 numerator **2.02e-4**；
per-coordinate CV median 6.88e-4，P90 1.96e-2。

**Q1 结论：** order64 coefficients 对 mask/λ 是**稳定坐标**；没有出现“reconstruction 1e-6 但 coefficient 变化 1e-2/1e-1”的现象。
complex probe 更稳定（8.5e-6），但这不是必要的替代坐标。





### 16.3 Full coefficients、per-output Ridge 与 PCA truncation（Q2/Q3）

Full 388-D standardized coefficients + per-output Ridge（λ_j 在 train 0:7000/7000:8000 上逐参数选择）：

```text
normalized MAE 0.18615
physical MAE   0.02430
mean R²        0.530
median R²      0.507
#R²<0.1        1  (param 23)
weak 21/23/26/31 R² = 0.136 / 0.080 / 0.174 / 0.151
```

PCA truncation（同样 per-output Ridge）：

| K | normalized MAE | physical MAE | mean R² | #R²<0.1 | 21 R² | 23 R² | 26 R² | 31 R² |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.23289 | 0.03010 | 0.349 | 5 | 0.076 | 0.043 | 0.126 | 0.089 |
| 128 | 0.21199 | 0.02758 | 0.427 | 2 | 0.100 | 0.040 | 0.128 | 0.110 |
| **256** | **0.18609** | **0.02429** | **0.530** | **1** | 0.140 | 0.073 | 0.186 | 0.152 |
| 320 | 0.18578 | 0.02426 | 0.532 | 1 | 0.141 | 0.079 | 0.184 | 0.150 |
| 388 (full) | 0.18608 | 0.02429 | 0.530 | 1 | 0.137 | 0.073 | 0.182 | 0.151 |

**Q2 结论：** PCA256 已经接近 full-rank；没有观察到某些 parameter 只在 full-rank 突然恢复。
PCA truncation **没有明显丢失 target-relevant information**。但 full coefficients 单独仍不如 Stage 6 的
`rational256 + raw PCA64`：

| feature | physical MAE | mean R² | #R²<0.1 |
|---|---:|---:|---:|
| full coefficients only | 0.02430 | 0.530 | 1 |
| full + raw64 | 0.02337 | 0.564 | 1 |
| **PCA256 + raw64** | **0.02330** | **0.564** | **1** |
| Stage 6 V4 Ridge reference | **0.02329** | **0.564** | **1** |

**Q3 结论：** full coefficient + per-output λ 没有超过 Stage 6；raw PCA64 仍是必要的稳定补充。

### 16.4 Supervised PLS2（Q4）

train-only PLS2 deflation，latent dimension → per-output Ridge：

| latent dim | normalized MAE | physical MAE | mean R² |
|---:|---:|---:|---:|
| 16 | 0.25975 | 0.03359 | 0.236 |
| 24 | 0.24738 | 0.03203 | 0.293 |
| 32 | 0.23843 | 0.03094 | 0.330 |
| 41 | 0.23179 | 0.03014 | 0.358 |
| 64 | 0.21912 | 0.02847 | 0.407 |
| 96 | 0.20549 | 0.02684 | 0.461 |
| 128 | 0.19861 | 0.02590 | 0.487 |

**Q4 结论：** supervised PLS **没有优于** unsupervised PCA256 / full coefficients；低维 supervised latent 不足，


### 16.5 Coeff group ablation（Q5）

| feature group | physical MAE | mean R² | weak 21/23/26/31 R² |
|---|---:|---:|---|
| G1 denominator only | 0.02868 | 0.398 | 0.095 / 0.062 / 0.169 / 0.131 |
| G2 S11 numerator only | 0.02697 | 0.443 | 0.126 / 0.082 / 0.180 / 0.149 |
| G3 S21 numerator only | 0.03490 | 0.177 | 0.059 / 0.018 / 0.075 / 0.012 |
| G4 D + S11 | 0.02557 | 0.491 | 0.132 / 0.090 / 0.187 / 0.167 |
| G5 D + S21 | 0.02716 | 0.447 | 0.097 / 0.056 / 0.158 / 0.129 |
| G6 all | **0.02430** | **0.530** | 0.136 / 0.080 / 0.174 / 0.151 |

**Q5 结论：** S11 numerator 是单个最强 group；denominator 次之；S21 numerator 单独最弱。
21/23/26/31 的信息几乎都已在 S11 numerator 中（G2 与 G6 的 R² 接近），S21 对这些参数帮助很小。
危险参数 5/8/10 也主要来自 S11；S21 对 35/37 有一定补充。若后续做结构模型，D+S11 是自然的 first branch，
S21 作为 secondary branch；但本阶段没有证据支持比 full 特征更强的 group-specific 结构。

### 16.6 Complex-plane probes（Q6）

固定 32 个实轴位置，η=0.02/0.05/0.10：

| probe | feature dim | normalized MAE | physical MAE | mean R² |
|---|---:|---:|---:|---:|
| P1 η=0.02 | 128 | 0.23271 | 0.02985 | 0.298 |
| P2 η=0.02+0.05 | 256 | 0.22848 | 0.02933 | 0.311 |
| P3 η=0.02+0.05+0.10 | 384 | 0.23122 | 0.02967 | 0.196 |
| P4 P3 + 1/D | 576 | **0.21653** | **0.02792** | 0.291 |

Probe stability 明显优于 coefficient：8.52e-6 vs 1.87e-4；但预测力明显更差（0.2165 vs full 0.1862）。

**Q6 结论：** complex probe 是更稳定的函数坐标，但不是更好的 inverse feature；没有带来参数回归收益。
P4 加 denominator 有所改善，但仍远差于 full coefficients。

### 16.7 Quadratic supervised inverse 与 residual structure（Q7）

在 PLS latent d=16/24/32 上构造 `[z, z², z_i z_j]`，再 per-output Ridge：

| latent d | quadratic MAE | quadratic physical MAE | linear PLS MAE（同 d） |
|---:|---:|---:|---:|
| 16 | 0.25170 | 0.03265 | 0.25975 |
| 24 | 0.23553 | 0.03067 | 0.24738 |
| 32 | 0.22342 | 0.02925 | 0.23843 |

Quadratic 相对同维 linear PLS 有小幅改善（d=32: 0.2384→0.2234），但仍远差于 full/PCA（0.186）。
Residual structure（base 0:5000，residual fit 5000:6500，eval 6500:8000）：

```text
base eval MAE       0.19018
combined eval MAE   0.20527
residual eval R²    mean -0.183, median -0.177
```

**Q7 结论：** 没有明显 quadratic 收益，residual 也不可预测；当前模型不是简单低阶非线性 underfit。

高维也只是接近 full，而不是超过它。target-relevant low-variance directions 没有带来可见增益。



### 16.8 One-shot（Q8）

Validation 1000，全部候选均为 0% pass：

| candidate | mean miss | median | P90 | worst |
|---|---:|---:|---:|---:|
| full coefficients | 20.40 dB | 18.66 | 29.92 | 61.39 |
| PCA320 | 20.16 dB | 18.67 | 28.46 | 48.35 |
| **PCA256 + raw64** | **19.83 dB** | **18.49** | **26.83** | **58.49** |
| PLS128 | 21.06 dB | 18.88 | 31.49 | 89.36 |
| probe P4 | 21.73 dB | 19.02 | 32.13 | 89.46 |
| quadratic d=32 | 22.02 dB | 19.16 | 34.55 | 61.37 |
| Stage 6 Global Ridge reference | 19.80 dB | 18.49 | 26.17 | 54.93 |
| Stage 6 ResMLP reference | **18.96 dB** | 18.13 | 19.59 | 54.83 |

**Q8 结论：** Stage 7 的最好 one-shot 是 `PCA256+raw64` 的 **19.83 dB**，没有超过 Stage 6
Global Ridge（19.80）或 ResMLP（18.96）；全部 pass=0。Stage 7 没有出现 one-shot 改善。

### 16.9 Q9 — 是否进入 canonical realization / Loewner？

Stage 7 的实际结果是：

```text
coefficient 稳定
PCA256 ≈ full-rank
supervised PLS < unsupervised PCA
complex probes 更稳定但不更有预测力
quadratic / residual 无增益
full coeff alone 不如 rational+raw
```

因此**没有证据**表明当前瓶颈是 rational feature extraction/compression 丢失了 inverse information；
也没有证据表明低阶 structured nonlinear inverse 能解决问题。基于 Stage 7，**不建议现在进入
canonical state-space / Loewner 路线**；那会是 speculative 的下一步，而不是由数据支持的结论。

### 16.10 Stop

- 已完成 stability audit、full coefficient regression、PCA truncation、PLS2、group ablation、
  complex probes、quadratic、residual structure、validation one-shot。
- **未新增任何 state，未使用新 seed/public seeds，未重优化 anchor，未训练 FNO/Transformer。**
- Stage 7 没有超过 Stage 6；停止。若未来继续，应先解决 one-shot 精度门槛与 compliance sensitivity，


---

## 17. Stage 8 — learned forward surrogate / numerical refinement gate

### 17.1 范围与 split

- 只使用已有 train 8000 / validation 1000；未 sample_state、未新增 seed、未 public seeds、未重优化 anchor。
- forward model 与 initializer 只用 **train 0:6999** 拟合；**train 7000:7999** 只做 architecture/early-stop/solver tuning。
- validation 未参与任何拟合或 solver 选择。
- Z1 = raw response PCA latent（K=256/512）；PCA 只用 train 0:6999 拟合，tune 7000:7999 只做投影。
- Z2 = rational PCA256 + raw PCA64（冻结的 Stage 6/7 latent，仅作研究诊断；其无监督变换在 train 8000 上拟合过，但未使用 tune/validation 标签）。
- 初始化器固定为 Z2 feature + Global Ridge，tune normalized MAE **0.18000**、physical MAE **0.02347**。

### 17.2 Forward surrogate（Q1）

| observation | model | latent RMSE | mean R² | median R² | P10 R² |
|---|---|---:|---:|---:|---:|
| Z1-256 | Linear | 0.991 | 0.057 | 0.00 | −0.01 |
| Z1-256 | ResMLP | 0.991 | 0.057 | −0.01 | −0.05 |
| Z1-512 | Linear | 1.041 | 0.027 | −0.01 | −0.01 |
| Z1-512 | ResMLP | 1.046 | 0.018 | −0.02 | −0.04 |
| Z2-320 | Linear | 0.896 | 0.189 | 0.136 | 0.025 |
| Z2-320 | ResMLP | **0.746** | **0.435** | 0.447 | 0.088 |

Z1 的 raw reconstruction（从 predicted PCA latent 重建）：

| K | reconstruction relL2 | dB median | dB P90 | spec error median | spec P90 |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.319 | 1.36 | 11.92 | 19.87 dB | 34.56 |
| 512 | 0.327 | 1.39 | 12.00 | 19.79 dB | 33.62 |

**Q1 结论：** 现有 forward surrogate **并不准确**。Z1 的 latent R² 只有 0.02–0.06，
raw reconstruction relL2 ≈0.32–0.33；Z2 的 ResMLP 好一些（R² 0.435），但仍不是可用的 forward model。
因此 Stage 8 的负结果**不能**直接解释为 inverse 难度，首先是 forward surrogate 本身没学好。

### 17.3 Surrogate floor（Q2）

| observation | r_floor median | r0 median | r0/r_floor median | P90 |
|---|---:|---:|---:|---:|
| Z1-256 | 15.34 | 15.24 | **0.994** | 1.012 |
| Z1-512 | 21.76 | 21.71 | **0.996** | 1.006 |
| Z2-320 | 11.37 | 11.91 | **1.044** | 1.160 |

**Q2 结论：** `r0 ≈ r_floor`，initializer 已经落在 learned surrogate 的误差底线上；
没有可用的 refinement signal。这是 Stage 8 最重要的 stop gate。

  而不是继续更换 feature compression。



### 17.4 Learned Jacobian quality（Q3）

参数空间 nearest pairs（256 tune states，δu norm median 2.245）：

| observation | cosine median | cosine P10 | relative L2 median | explained fraction median |
|---|---:|---:|---:|---:|
| Z1-512 | 0.132 | 0.082 | 0.994 | 0.011 |
| Z2-320 | 0.399 | 0.254 | 1.102 | −0.216 |

**Q3 结论：** learned Jacobian 不能预测真实 local `Δz`；Z1 基本无方向一致性，Z2 也只有中等 cosine，
explained fraction 接近 0 或为负。Jacobian 不足以支持 LM/refinement。

### 17.5 Adam / LM refinement（Q4/Q5）

Initializer：normalized MAE **0.18000**，physical MAE **0.02347**，mean R² 0.567。

| observation | solver | best setting | true normalized MAE | physical MAE | mean R² |
|---|---|---|---:|---:|---:|
| Z1-512 | Adam | steps5, lr1e-2 | 0.18366 | 0.02415 | 0.559 |
| Z1-512 | LM | iter1, damp1 | 0.21366 | 0.02807 | 0.368 |
| Z1-512 | Adam weighted | steps5, lr1e-2 | 0.18293 | 0.02404 | 0.562 |
| Z2-320 | Adam | steps5, lr1e-2 | 0.17770 | 0.02330 | 0.585 |
| Z2-320 | LM | iter1, damp1 | 0.17993 | 0.02340 | 0.538 |
| Z2-320 | Adam weighted | steps5, lr1e-2 | **0.17768** | **0.02329** | **0.585** |

- Z1 的 refinement **变差**；LM 更差。
- Z2 的 Adam 仅有 **0.00015 physical MAE** 的改善，远未达到 Stage 8 的 `<0.018` gate。
- 加权只是为了很小改善；没有出现 `0.023 → <0.018` 的跃迁。

**Q4/Q5 结论：Adam 与 LM 都没有降低 true state error 到有意义的程度；LM 更不稳定。**

### 17.6 Surrogate exploitation（Q6）

| observation | r0 median | r_ref median | true MAE initial → refined | exploitation |
|---|---:|---:|---|---|
| Z1-512 | 21.71 | 21.49 | 0.18000 → 0.18366 | **True** |
| Z2-320 | 11.91 | 10.83 | 0.18000 → 0.17770 | False |

Z1 出现典型的 surrogate exploitation：surrogate residual 略降，但 true parameter error 变差。
Z2 没有触发 flag，但真实改善极小，不能视为成功。

### 17.7 Weak / dangerous parameters（Q7）

Z2-320 Adam 的 weak R²：
`21 0.155, 23 0.213, 26 0.147, 31 0.248`，相对 initializer
`0.141/0.190/0.139/0.239` 只有极小变化。  
Z1 refinement 的 weak R² 反而下降。Dangerous parameters 也没有出现可用改善；
physical dangerous MAE/P95 仍在 0.023–0.028 / 0.067–0.078 量级。

### 17.8 One-shot（Q8）

Stage 8 gate 未通过（best physical MAE 0.02329，远高于 0.018），因此**没有**重训 full-8000
forward/initializer，也没有跑 validation one-shot。当前 best one-shot 仍是 Stage 6：

```text
ResMLP: pass 0/1000, mean 18.96 dB, median 18.13, P90 19.59, worst 54.83
```

**Q8：** 不适用/未执行；没有新的 one-shot 结果，不能声称 learned-forward refinement 改善了 control。

### 17.9 Q9 — 是否值得进入 learned unrolled refinement？

**不值得。** Stage 8 的 gate 是：

```text
forward surrogate 不够准确
r0 / r_floor ≈ 1
Jacobian 不具备局部预测力
Adam/LM true state error 不降
Z1 出现 surrogate exploitation
```

因此不应训练 `u_{k+1}=u_k+Rθ(u_k,z_obs-G(u_k))`，也不应把 learned-forward inversion 作为下一阶段默认方向。
这不是“inverse 不可能”的证明；更准确的结论是：**当前 learned forward surrogate 本身不足以提供 refinement signal**。

### 17.10 Stop

- 已完成 Z1/Z2 forward surrogate、surrogate floor、Jacobian quality、Adam、LM、true-state evaluation。
- 未进行 full-validation one-shot（gate 未通过）。
- **未新增任何 state，未使用新 seed/public seeds，未重优化 anchor，未训练 FNO/Transformer。**


---

## 18. Stage 9 — smooth specification objective + zero-order residual policy

### 18.1 范围与 base policy

- base：已有 TorchScript-compatible Route 3 raw+Tukey FNO m32（`runs/route3_modes/r3_m32_s2026/best.pt`）。
- 只使用已有 train 8000 / validation 1000；模拟器只对 cached `m_current` 的 candidate corrected vector 调用。
- 没有新增 state / seed / public seed；没有重优化 anchor；没有 FNO/Transformer 训练。
- 训练 split：tune 阶段 train 0:6999 / tune 7000:7999；positive gate 后用 full 0:7999 重训 residual head。
- residual head zero-init；base encoder 完全冻结。`Δm = Δm_base + r ⊙ δu`。

### 18.2 Smooth objective calibration（Q1）

64 个 tune states；smooth seven-spec loss vs 真实 `max_spec_miss_db`：

| tau | beta | tau_w | Pearson | Spearman |
|---:|---:|---:|---:|---:|
| **0.10** | **0.50** | **0.25** | **0.9865** | **0.9543** |

Base 在这 64 个 tune states：mean 20.40 dB、median 18.66、P90 25.74、pass 0。

**Q1：** smooth loss 与 hard miss 高度相关，objective 可用于训练。

### 18.3 Coordinate gradient / oracle（Q2/Q3）

16 个 tune states，coordinate finite difference：

| eps | gradient norm mean | norm P90 |
|---:|---:|---:|
| 0.005 | 22.84 | 35.87 |
| 0.010 | 34.63 | 98.57 |
| 0.020 | 46.19 | 130.07 |

不同 eps 的 gradient cosine median **0.99998**（P10 0.9972）→ 极稳定。

One-step exact coordinate-gradient oracle（g010）：

| eta | mean miss | median | P90 | improved fraction |
|---:|---:|---:|---:|---:|
| base | 20.01 | 18.58 | 19.55 | — |
| **0.05** | **17.79** | **17.53** | **19.99** | **0.625** |
| 0.10 | 25.47 | 18.31 | 42.69 | 0.438 |
| 0.20 | 32.28 | 24.45 | 53.24 | 0.188 |
| 0.50 | 50.44 | 50.43 | 68.89 | 0.063 |

**Q2：coordinate gradient 稳定。**  
**Q3：一步 oracle 有真实下降方向，但步长必须小（eta=0.05）。**

### 18.4 SPSA fidelity（Q4）

16 states，eps=0.01：

| K | cosine median | cosine P10 | sign agreement | top-5 overlap |
|---:|---:|---:|---:|---:|
| 4 | 0.245 | 0.125 | 0.561 | 0.2 |
| 8 | 0.428 | 0.319 | 0.646 | 0.2 |
| **16** | **0.539** | **0.441** | **0.659** | **0.4** |

**Q4：** K=16 刚超过最低 cosine gate（>0.5），因此用于 residual-head 训练；K=4/8 不足。

- Stage 8 停止；不应把“更多数据”写成 recommendation。



### 18.5 Residual-head simulator-in-loop（Q5/Q6/Q7）

200 optimizer steps，batch=16，K=16，eps=0.01，lr=1e-4；AdamW；λ_bound=λ_trust=1e-3。

| split | base mean | refined mean | median | P90 | worst | pass |
|---|---:|---:|---:|---:|---:|---:|
| tune 1000 | 20.38 | **13.70** | 13.84 | 16.02 | 19.39 | 0 |
| validation 1000 | 21.56 | **14.66** | 14.83 | 16.74 | 18.43 | 0 |

Tune step-200 subset-64：mean 13.54、P90 15.13、pass 0。  
Validation gate：**mean < 17 dB 通过；P90 < 20 dB 通过；pass 仍为 0。**

Per-requirement validation：

| requirement | base pass | refined pass | refined mean miss |
|---|---:|---:|---:|
| S11_1 | 0.0% | **0.0%** | 13.37 dB |
| S11_2 | 0.0% | **0.0%** | 13.44 dB |
| S21_1 | 6.9% | 80.4% | 0.12 dB |
| S21_2 | 0.0% | 41.4% | 0.40 dB |
| S21_3 | 9.7% | 62.0% | 0.26 dB |
| S21_4 | 0.1% | 36.6% | 0.60 dB |
| S21_5 | 7.2% | 66.8% | 0.27 dB |

**Q5：** refined mean 从 20.38→13.70（tune）、21.56→14.66（validation），median/P90 同步大幅下降；  
**Q6：** 仍然 **pass=0**。  
**Q7：** S11 是唯一剩余瓶颈；S21 已大幅改善，两条 S11 requirement 仍 0% pass、mean miss 约 13.4 dB。

Residual magnitude（validation）：mean |δu| **0.862**，P90 **1.702**；tune 为 1.294 / 2.698。  
Specification-gradient top parameters（base coordinate gradient）：**19, 26, 25, 17, 16, 28, 7, 21, 20, 18**。



### 18.6 Q8 — best validation one-shot

Residual-head full-8000 model on validation 1000：

```text
pass        0/1000 (0%)
mean miss   14.66 dB
median      14.83 dB
P90         16.74 dB
worst       18.43 dB
```

这比 Stage 6 ResMLP（mean 18.96、P90 19.59）明显更好，但仍没有 all-seven pass；
剩余 miss 完全由 S11 的两个 requirement 决定。

### 18.7 Q9 — 是否解冻 encoder？

Residual-head positive gate 已通过（mean<17、P90<20），因此根据 Stage 9 规则，
**partial encoder unfreeze 是允许的**；但本阶段**没有执行**，因为当前 evidence 指向更具体的方向：

- 残余误差集中在 **S11**；S21 已基本解决；
- generic unfreeze 很可能继续优化已经较好的 S21，而不是 S11；
- 更合理的第一步是 **S11-weighted residual objective**（λ_S11>1）或 S11-specific head，
  再考虑 only last FNO block + aggregation + residual head 的 controlled unfreeze。

**Q9 结论：** 值得做的是 **S11-targeted residual / controlled last-stage unfreeze**，  
而不是全 encoder unfreeze；本阶段在 residual-head positive gate 后停止。

### 18.8 Simulator-call accounting

Core final artifacts：

| phase | calls |
|---|---:|
| smooth calibration (64 states) | 64 |
| coordinate gradient (16 states × 3 eps × 82) | 3,936 |
| one-step oracle (16 × (1+4)) | 80 |
| SPSA fidelity (16 × (4+8+16) × 2) | 896 |
| tune residual training (200 × 512 + evals) | 104,656 |
| tune analysis | 2,000 |
| full residual training (200 × 512 + evals) | 104,656 |
| validation analysis | 2,000 |
| **total core** | **≈218,288** |

另有 2-step smoke / 早期探索约 2k calls；未计入 core 表。没有新增 simulator state。

### 18.9 Stop

- 已完成 smooth calibration、coordinate gradient、oracle、SPSA fidelity、residual-head tune/full training、validation one-shot。
- **未新增 state / seed / public seed；未重优化 anchor；未训练 FNO/Transformer。**
- 没有执行 partial encoder unfreeze；本阶段在 residual-head positive gate 后停止。


---

## 19. Stage 10 — S11-targeted gate（clipping / gradient / oracle / head）

### 19.1 范围与起点

- 起点：Stage 9 full-8000 refined policy（base FNO m32 + frozen general residual head）。
- 只使用 train-tune 7000:7999；没有 validation 拟合；没有新增 state / seed / public seed / anchor 搜索。
- 代码：`scripts/stage10_study.py`；结果：`artifacts/stage10/*.json`。

### 19.2 Clipping / bound saturation（Q1）

256 个 tune states：

| metric | value |
|---|---:|
| mean |y| | 1.014 |
| P90 |y| | 2.062 |
| mean L2 | 8.03 |
| outside-coordinate fraction（clip 前） | **0.4679** |
| samples with ≥1 clipped coordinate | **1.000** |
| S11 high-miss group outside rate | 0.4688 |
| S11 low-miss group outside rate | 0.4670 |

参数 10/11/12/15/20/25/35 的 clipping rate 接近 **100%**；lower-bound 最常撞的是 11/15/25/36，
upper-bound 最常撞的是 10/12/20/35/28。

**Q1：Stage 9 policy 高度依赖 clipping。** 这不是少见的边界情况，而是主要工作方式之一；
高 S11 miss 与低 S11 miss 的 clipping 程度几乎相同。

### 19.3 Exact per-requirement gradients（Q2/Q3/Q4）

32 个 tune states，eps=0.01，从 Stage 9 refined y 出发：

| gradient | median norm | P10 | P90 |
|---|---:|---:|---:|
| S11_1 | 11.91 | 7.75 | 20.97 |
| S11_2 | 14.69 | 9.34 | 30.31 |
| **S11 (worst-of-two)** | **9.63** | 7.47 | 15.78 |
| **S21 barrier** | **4.53** | 2.11 | 8.43 |
| full | 12.66 | 9.12 | 19.00 |

**Q2：S11 gradient 不弱，反而大于 S21 barrier（9.63 vs 4.53）。**  
所以 Stage 9 没有改善 S11 不是“S11 gradient 太小”。

Gradient conflict：

| pair | cosine median | P10 | fraction <0 |
|---|---:|---:|---:|
| S11 vs S21 barrier | **0.872** | 0.685 | **0.0** |
| S11_1 vs S11_2 | **0.045** | −0.181 | 0.344 |

**Q3：S11/S21 基本协同，没有严重冲突。**  
**Q4：两个 S11 band 轻度冲突（34% cosine<0，但没有 <−0.5）。**

S11 gradient top parameters：

```text
23, 22, 21, 33, 31, 32, 38, 14, 18, 19, 4, 39, 6, 16, 30
```

这与 Stage 9 full-objective top（19,26,25,17,16,28,7,21,20,18）不同；
S11 的主要控制方向确实不是 full objective 的主要方向。

### 19.4 S11 gradient rank（Q5）

S11 gradient matrix `G11 ∈ R^{32×41}` 的 SVD：

```text
d=1  33.6%    d=4  75.5%    d=8  94.1%
d=9  95.7%    d=12 98.2%    d=16 99.5%
```

95% energy rank：**9**。因此 `B11` 取 **d=9**。

**Q5：S11 action space 非常低维，9 维足够解释 95%。**

### 19.5 Exact oracle（Q6/Q7）

从 Stage 9 y 出发，line search（32 states）：

| direction | eta | S11 mean | overall mean | S21 mean |
|---|---:|---:|---:|---:|
| full41 | 0.02 | 10.86 | 11.98 | 0.112 |
| B11 | 0.02 | 11.08 | 12.26 | 0.165 |

B11 与 full41 非常接近；**Q6：低维 B11 保留了 full-41 的大部分 oracle 效果。**

Multi-step exact oracle（16 states，full41，eta=0.02）：

| step | S11 mean | overall mean | S21 mean |
|---:|---:|---:|---:|
| 0 | 12.62 | 14.19 | 0.634 |
| 1 | 11.34 | 12.47 | 0.182 |
| 2 | 9.90 | 11.23 | 0.068 |
| 4 | 8.48 | 9.81 | 0.001 |
| 8 | **8.10** | **9.34** | **0.025** |

**Q7：存在持续下降的局部路径；S11 从 12.6→8.10，overall 9.34，S21 没有被破坏。**
Gate A（<8）和 B（S11 pass>0）未达到；Gate C/D 达到（overall<10 且持续下降）。

### 19.6 Low-dimensional SPSA（Q8）

d=9 的 B11 subspace，16 states：

| K | cosine median | cosine P10 | sign agreement |
|---:|---:|---:|---:|
| 2 | 0.504 | 0.272 | 0.667 |
| 4 | 0.596 | 0.458 | 0.667 |
| **8** | **0.750** | **0.622** | 0.667 |

**Q8：低维 SPSA 明显优于 Stage 9 的 0.539；K=8 达到 0.75。**

### 19.7 S11-specific head（Q9）

S11 head：Stage 9 encoder latent → Linear 256/GELU/ResFC → `c∈R^9`，zero-init；
`y_final = y_Stage9 + B11 c`。Stage 9 model 与 B11 冻结。

200 steps，batch 16，K=8，eps=0.01，lr=1e-4：

| split | base S11 | refined S11 | base overall | refined overall | pass |
|---|---:|---:|---:|---:|---:|
| tune 1000 | 12.87 | **12.69** | 14.15 | **13.98** | 0 |

`c` mean |c| **0.062**，P90 0.147；S11 head 只学到很小的动作，S11 几乎没有改善。
**Q9：没有出现 S11 pass >0；S11-head positive gate 失败。**

### 19.8 Q10 — 是否 partial encoder unfreeze？

**不允许进入。** Stage 10 规则是只有 S11-head positive gate 通过才允许 partial unfreeze；
本阶段 S11 head 失败，因此没有解冻 encoder，也没有跑 validation full head。

### 19.9 Simulator-call accounting

| phase | calls |
|---|---:|
| clipping audit | 256 |
| exact gradients | 2,656 |
| exact oracle | 10,848 |
| low-D SPSA fidelity | 448 |
| S11 head training | 53,456 |
| **total** | **≈67,664** |

### 19.10 Stop

- 已完成 clipping audit、per-requirement gradients、conflict、SVD、full/B11 oracle、low-D SPSA、S11 head。
- **未新增 state / seed / public seed；未重优化 anchor；未训练 FNO/Transformer。**
- Stage 10 结论：S11 确实有低维、可下降的 exact-gradient 路径，但当前以 Stage 9 latent 为输入的
  S11 head 没有学会足够大的动作；不应直接 partial unfreeze。
- 下一步若要继续，应优先解决 **S11 head 的优化/幅度问题**（或 S11-specific features），
  而不是增加数据或换 representation。


---

## 20. 2026-11-18 封存与恢复清单

### 20.1 一句话状态

> Stage 1–10 已完成；Stage 9 的 refined policy 是当前最佳（validation **14.66 dB / pass 0**）；
> Stage 10 证明 S11 有低维 exact-gradient 下降路径，但 S11-specific head 失败；
> **现在停止，不解冻 encoder，等待新的明确指令。**

### 20.2 本次真正新增的东西

| 类型 | 文件 | 说明 |
|---|---|---|
| 代码 | `scripts/stage10_study.py` | `audit / gradients / oracle / lowd / s11head / s11headfull / all` |
| 结果 | `artifacts/stage10/clipping_audit.json` | Q1 |
| 结果 | `artifacts/stage10/gradient_analysis.json` | Q2–Q5 |
| 结果 | `artifacts/stage10/oracle.json` | Q6–Q7 |
| 结果 | `artifacts/stage10/lowd_spsa.json` | Q8 |
| 结果 | `artifacts/stage10/s11_head_training.json` | Q9 |
| 文档 | `HANDOVER.md` §19/§20 | Stage 10 详情 + 本清单 |
| 文档 | `README.md` | Stage 10 摘要 |

本地生成、**未提交**（git-ignored）：

```text
artifacts/stage10/s11_gradients.npz
artifacts/stage10/s11_head.pt
artifacts/stage9/residual_head_full.pt
runs/...
cache/...
```

新机器/新 clone 只有 JSON 结果，没有 `.pt/.npz`；需要时用脚本重跑或从本机拷贝。
不要把 `/tmp/*.log` 当成交付物。

### 20.3 当前结果的准确表述

- Stage 9 positive gate：tune mean 20.38→13.70，validation 21.56→**14.66 dB**（P90 16.74），
  但 **pass 0/1000**；S21 基本解决，**两个 S11 requirements 均 0% pass**。
- Stage 10 oracle gate：
  - 单步 full41 / B11：S11 **10.86 / 11.08**（32 states，eta=0.02）；
  - 8-step full41：S11 **12.62→8.10**，overall **14.19→9.34**，S21 **0.634→0.025**；
  - Gate C/D 通过；Gate A（S11<8）与 B（S11 pass>0）没有通过。
- Stage 10 S11 head：
  - tune 1000：S11 **12.87→12.69**，overall **14.15→13.98**，pass **0**；
  - 动作幅度过小：`mean |c| = 0.062`，P90 0.147；
  - **negative gate**；因此没有 full-8000 head、没有 validation、没有 unfreeze。
- Stage 10 simulator calls：**67,664**（明细见 §19.9）。

### 20.4 新对话的下一步（按优先级）

**不要直接跑 `s11headfull`，也不要直接 partial unfreeze。**

1. **先诊断 S11 head 为什么动作小（推荐）**
   - 检查 `stage_s11head()` 的 loss 各项、`c` 的分布、梯度是否被 trust/边界项压制；
   - 仍只使用 train-tune **7000:7999**；
   - 可试一个受控变体（只选一个）：
     - lr `3e-4` + 300 steps；或
     - 去掉/降低 `lam_trust`，检查 `c` 是否能自然增大；或
     - 用 S11-specific feature（例如 per-band margin / band-wise pooling），但不要换 FNO。
   - 通过条件仍是原 gate：S11 mean 明显下降，或出现 **S11 pass>0**。
2. **只有 head gate 通过后**：
   - 跑 `s11headfull`（full 0:7999）；
   - validation 1000:1999 **只评估一次**；
   - 冻结后再考虑 internal test 30000–30999。
3. **只有 full head 仍正、且明确需要时**：
   - 才考虑 controlled partial unfreeze（last FNO block + aggregation + S11 head；
     encoder lr = 0.1×head lr）；
   - 必须重新走 gate，不可直接上 validation。
4. 如果不想继续模型研究：直接进入报告/答辩材料整理，保留 Stage 9 结果作为最终 policy。

### 20.5 复现命令

```bash
cd /home/qian/DL_assignment_2026
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 只跑已完成的单阶段（按需）
python scripts/stage10_study.py audit
python scripts/stage10_study.py gradients
python scripts/stage10_study.py oracle
python scripts/stage10_study.py lowd
python scripts/stage10_study.py s11head

# 禁止直接跑 all：all 会包含 s11headfull
```

注意：

- `all` 会执行 `s11headfull`，当前 gate 未通过，**不要用**。
- 多进程 simulator 必须设置单线程 BLAS，否则会线程超订。
- GitHub 推送需要代理 `http://127.0.0.1:7890`。
- `scripts/stage10_study.py` 的 S11 head 超参目前是函数内硬编码；
  若要试 lr/步数变体，先改函数参数，避免覆盖现有 `s11_head_training.json` 前没有备份。

### 20.6 不可违反的约束（再次强调）

- 不新增 simulator state / seed；不使用 public seeds 10000–10049。
- 不重新优化 anchor。
- 不训练 FNO / Transformer / 新 representation；本阶段只做 S11 head/诊断。
- 不在 validation 上拟合或反复调参；validation 只在冻结后评估一次。
- 没有 S11-head positive gate，不做 partial encoder unfreeze。
- 不把 `runs/`、`cache/`、`.pt`、`.npz` 误提交；JSON 结果才是可移植交付物。

### 20.7 给新对话的起始提示

可以直接把下面这段发给新对话：

```text
继续 CM19 项目。先读 HANDOVER.md 的「立即恢复」、§18、§19、§20。
当前 Stage 10 已封存：Stage 9 policy 是当前最佳（validation 14.66 dB，pass 0）；
Stage 10 S11 head 失败（S11 12.69，pass 0，c 幅度仅 0.062），所以没有 full head、
没有 unfreeze。不要跑 s11headfull，不要重优化 anchor，不要新增 state/seed。
下一步只在 train-tune 7000:7999 诊断 S11 head 的动作幅度/优化问题。
```
