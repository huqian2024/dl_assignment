"""Minimal, reusable training framework (framework only, no long schedules).

Features: AdamW, AMP, gradient clipping, cosine schedule, validation, best
checkpoint selection, early stopping, JSONL logging, physical error metrics.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluator import evaluate_one_shot
from .losses import state_error_metrics, state_loss
from .utils import (
    RUNS_DIR,
    append_jsonl,
    count_parameters,
    get_device,
    resolve,
    save_json,
    set_seed,
)


def _make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _amp_objects(device: torch.device, enabled: bool, dtype: torch.dtype):
    use_amp = bool(enabled and device.type == "cuda")
    # bf16 keeps the fp32 exponent range, so no loss scaling is needed.
    scale_enabled = bool(use_amp and dtype == torch.float16)
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=scale_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=scale_enabled)
    return use_amp, scaler


def _autocast(device: torch.device, enabled: bool, dtype: torch.dtype):
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _resolve_amp_dtype(name: str) -> torch.dtype:
    key = str(name).lower()
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unknown amp dtype '{name}'")


class Trainer:
    """Runs one training configuration; deliberately small and explicit."""

    def __init__(
        self,
        model: torch.nn.Module,
        model_cfg: dict,
        train_cfg: dict,
        train_dataset,
        val_dataset=None,
        output_dir: str | None = None,
        run_name: str | None = None,
        device: str | torch.device = "auto",
        simulator=None,
        one_shot_seeds=None,
    ) -> None:
        self.device = get_device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.model_cfg = dict(model_cfg)
        self.cfg = dict(train_cfg)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.simulator = simulator
        self.one_shot_seeds = (
            [int(seed) for seed in one_shot_seeds] if one_shot_seeds is not None else None
        )
        self.run_name = run_name or f"run_{int(time.time())}"
        base = Path(resolve(output_dir or self.cfg.get("output_dir", RUNS_DIR)))
        self.output_dir = base / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "log.jsonl"
        self.best_path = self.output_dir / "best.pt"
        self.best_loss_path = self.output_dir / "best_loss.pt"
        self.best_mae_path = self.output_dir / "best_mae.pt"
        self.best_pass_path = self.output_dir / "best_pass.pt"
        self.last_path = self.output_dir / "last.pt"
        self.simulator_calls = 0
        set_seed(int(self.cfg.get("seed", 2026)))
        self._build_optimizer()

    # -- setup -----------------------------------------------------------
    def _build_optimizer(self) -> None:
        cfg = self.cfg
        params = [p for p in self.model.parameters() if p.requires_grad]
        if str(cfg.get("optimizer", "adamw")).lower() != "adamw":
            raise ValueError("Only AdamW is implemented in this stage")
        self.optimizer = torch.optim.AdamW(
            params,
            lr=float(cfg.get("learning_rate", 3e-4)),
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
        )
        self.train_loader = _make_loader(
            self.train_dataset,
            int(cfg.get("batch_size", 32)),
            shuffle=True,
            num_workers=int(cfg.get("num_workers", 0)),
        )
        self.val_loader = (
            _make_loader(
                self.val_dataset,
                int(cfg.get("batch_size", 32)),
                shuffle=False,
                num_workers=int(cfg.get("num_workers", 0)),
            )
            if self.val_dataset is not None
            else None
        )
        steps_per_epoch = max(1, len(self.train_loader))
        total_steps = max(1, steps_per_epoch * int(cfg.get("epochs", 1)))
        warmup = int(cfg.get("warmup_steps", 0))
        self.total_steps = total_steps
        self.warmup_steps = warmup
        self.amp_dtype = _resolve_amp_dtype(self.cfg.get("amp_dtype", "bf16"))
        self.use_amp, self.scaler = _amp_objects(
            self.device, bool(cfg.get("amp", True)), self.amp_dtype
        )

    def _scheduler_step(self) -> None:
        schedule = str(self.cfg.get("scheduler", "cosine")).lower()
        step = self.step
        if schedule in ("none", ""):
            factor = 1.0
        elif schedule == "cosine":
            warm = min(1.0, (step + 1) / max(1, self.warmup_steps)) if self.warmup_steps else 1.0
            progress = min(1.0, step / max(1, self.total_steps))
            factor = warm * (0.5 * (1.0 + math.cos(math.pi * progress)))
        else:
            raise ValueError(f"Unknown scheduler '{schedule}'")
        for group in self.optimizer.param_groups:
            group["lr"] = float(self.cfg.get("learning_rate", 3e-4)) * factor

    # -- epochs ----------------------------------------------------------
    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_states = 0
        start = time.perf_counter()
        for response, m_current in self.train_loader:
            response = response.to(self.device, non_blocking=True)
            m_current = m_current.to(self.device, non_blocking=True)
            target_u = self.model.state_normalizer.encode(m_current)
            self.optimizer.zero_grad(set_to_none=True)
            with _autocast(self.device, self.use_amp, self.amp_dtype):
                pred_u, _, _ = self.model.forward_state(response)
                loss = state_loss(
                    pred_u,
                    target_u,
                    kind=str(self.cfg.get("loss", "huber")),
                    delta=float(self.cfg.get("huber_delta", 1.0)),
                )
            self.scaler.scale(loss).backward()
            clip = float(self.cfg.get("grad_clip", 1.0))
            if clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.step += 1
            self._scheduler_step()
            total_loss += float(loss.detach()) * int(response.shape[0])
            total_states += int(response.shape[0])
        elapsed = time.perf_counter() - start
        return {
            "epoch": epoch,
            "split": "train",
            "loss": total_loss / max(1, total_states),
            "states": total_states,
            "seconds": elapsed,
            "states_per_second": total_states / max(elapsed, 1e-9),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, Any]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        total_loss = 0.0
        total_states = 0
        predictions = []
        targets = []
        start = time.perf_counter()
        for response, m_current in self.val_loader:
            response = response.to(self.device, non_blocking=True)
            m_current = m_current.to(self.device, non_blocking=True)
            target_u = self.model.state_normalizer.encode(m_current)
            with _autocast(self.device, self.use_amp, self.amp_dtype):
                pred_u, pred_m, _ = self.model.forward_state(response)
                loss = state_loss(
                    pred_u,
                    target_u,
                    kind=str(self.cfg.get("loss", "huber")),
                    delta=float(self.cfg.get("huber_delta", 1.0)),
                )
            total_loss += float(loss.detach()) * int(response.shape[0])
            total_states += int(response.shape[0])
            predictions.append(pred_m.detach().float().cpu())
            targets.append(m_current.float().cpu())
        elapsed = time.perf_counter() - start
        metrics = state_error_metrics(
            torch.cat(predictions, dim=0), torch.cat(targets, dim=0)
        )
        return {
            "epoch": epoch,
            "split": "validation",
            "loss": total_loss / max(1, total_states),
            "states": total_states,
            "seconds": elapsed,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "median_abs": metrics["median_abs"],
            "p90_abs": metrics["p90_abs"],
            "p95_abs": metrics["p95_abs"],
            "p99_abs": metrics["p99_abs"],
            "max_abs": metrics["max_abs"],
            "l2_mean": metrics["l2_mean"],
            "l2_p95": metrics["l2_p95"],
            "l2_max": metrics["l2_max"],
            "linf_mean": metrics["linf_mean"],
            "linf_p95": metrics["linf_p95"],
            "linf_max": metrics["linf_max"],
            "per_parameter_mae": metrics["per_parameter_mae"].tolist(),
            "per_parameter_p95": metrics["per_parameter_p95"].tolist(),
        }

    def _one_shot(self, epoch: int, force: bool = False) -> dict[str, Any]:
        """Periodic official one-shot evaluation on a fixed validation subset."""
        every = int(self.cfg.get("one_shot_every", 0) or 0)
        if every <= 0 or self.simulator is None or not self.one_shot_seeds:
            return {}
        start_epoch = int(self.cfg.get("one_shot_start_epoch", every))
        if not force and (epoch % every != 0 or epoch < start_epoch):
            return {}
        if epoch < 0:
            return {}
        summary, _ = evaluate_one_shot(
            self.model,
            self.simulator,
            self.one_shot_seeds,
            device=self.device,
            batch_size=int(self.cfg.get("one_shot_batch_size", 16)),
        )
        self.simulator_calls += 2 * int(summary["num_states"])
        return {
            "epoch": epoch,
            "states": summary["num_states"],
            "pass_count": summary["pass_count"],
            "pass_rate": summary["pass_rate"],
            "mean_max_miss_db": summary["mean_max_miss_db"],
            "median_max_miss_db": summary["median_max_miss_db"],
            "p90_max_miss_db": summary["p90_max_miss_db"],
            "worst_max_miss_db": summary["worst_max_miss_db"],
            "correction_l2_mean": summary["correction_l2_mean"],
            "clipped_entry_rate_mean": summary["clipped_entry_rate_mean"],
        }

    def _save_checkpoint(self, path: Path, epoch: int, metrics: dict) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": int(epoch),
                "metrics": metrics,
                "val_loss": float(metrics.get("val_loss", float("nan"))),
                "model_cfg": self.model_cfg,
                "train_cfg": self.cfg,
            },
            path,
        )

    def fit(self) -> dict[str, Any]:
        self.step = 0
        best_loss = float("inf")
        best_mae = float("inf")
        best_pass = -1.0
        best_pass_miss = float("inf")
        best_epochs = {"loss": -1, "mae": -1, "pass": -1}
        patience = int(self.cfg.get("early_stopping_patience", 0))
        early_metric = str(self.cfg.get("early_stopping_metric", "mae")).lower()
        epochs_without_improvement = 0
        history = []
        fit_start = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        save_json(
            self.output_dir / "config.json",
            {
                "model_cfg": self.model_cfg,
                "train_cfg": self.cfg,
                "trainable_parameters": count_parameters(self.model),
                "one_shot_states": len(self.one_shot_seeds or []),
            },
        )
        total_epochs = int(self.cfg.get("epochs", 1))
        last_epoch = 0
        for epoch in range(1, total_epochs + 1):
            last_epoch = epoch
            train_record = self._train_epoch(epoch)
            val_record = self._validate(epoch)
            one_shot = self._one_shot(epoch)
            record = {
                "train": train_record,
                "validation": val_record or None,
                "one_shot": one_shot or None,
                "gpu_memory_mb": (
                    float(torch.cuda.max_memory_allocated() / 2**20)
                    if self.device.type == "cuda"
                    else None
                ),
            }
            history.append(record)
            append_jsonl(self.log_path, record)

            val_loss = (
                float(val_record.get("loss", train_record["loss"]))
                if val_record
                else float(train_record["loss"])
            )
            val_mae = (
                float(val_record.get("mae", float("inf")))
                if val_record
                else float("inf")
            )
            metrics_for_ckpt = {
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_p95_abs": (
                    float(val_record.get("p95_abs", float("nan")))
                    if val_record
                    else float("nan")
                ),
                "epoch": epoch,
            }
            self._save_checkpoint(self.last_path, epoch, metrics_for_ckpt)
            if val_loss < best_loss:
                best_loss = val_loss
                best_epochs["loss"] = epoch
                self._save_checkpoint(
                    self.best_loss_path,
                    epoch,
                    {**metrics_for_ckpt, "criterion": "val_loss"},
                )
                # legacy alias kept for earlier scripts
                self._save_checkpoint(
                    self.best_path,
                    epoch,
                    {**metrics_for_ckpt, "criterion": "val_loss"},
                )
            if val_mae < best_mae:
                best_mae = val_mae
                best_epochs["mae"] = epoch
                self._save_checkpoint(
                    self.best_mae_path,
                    epoch,
                    {**metrics_for_ckpt, "criterion": "val_mae"},
                )
            if one_shot:
                pass_rate = float(one_shot["pass_rate"])
                mean_miss = float(one_shot["mean_max_miss_db"])
                better = pass_rate > best_pass or (
                    pass_rate == best_pass and mean_miss < best_pass_miss
                )
                if better:
                    best_pass = pass_rate
                    best_pass_miss = mean_miss
                    best_epochs["pass"] = epoch
                    self._save_checkpoint(
                        self.best_pass_path,
                        epoch,
                        {
                            **metrics_for_ckpt,
                            "criterion": "one_shot_pass_rate",
                            "pass_rate": pass_rate,
                            "mean_max_miss_db": mean_miss,
                        },
                    )

            print(
                f"[train] epoch {epoch}/{total_epochs} "
                f"train_loss={train_record['loss']:.5f} "
                f"val_loss={val_loss:.5f} val_mae={val_mae:.5f} "
                f"val_p95={float(val_record.get('p95_abs', float('nan'))):.5f}"
                + (
                    f" pass={one_shot['pass_rate']:.3f} "
                    f"miss_mean={one_shot['mean_max_miss_db']:.2f}"
                    if one_shot
                    else ""
                ),
                flush=True,
            )

            monitored = val_mae if early_metric == "mae" else val_loss
            best_monitored = best_mae if early_metric == "mae" else best_loss
            if monitored <= best_monitored:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if patience and epochs_without_improvement >= patience:
                print(f"[train] early stopping at epoch {epoch}", flush=True)
                break
        if last_epoch > 0:
            final_one_shot = self._one_shot(last_epoch, force=True)
            if final_one_shot:
                append_jsonl(
                    self.log_path,
                    {"final_one_shot": final_one_shot, "epoch": last_epoch},
                )
                if final_one_shot["pass_rate"] > best_pass or (
                    final_one_shot["pass_rate"] == best_pass
                    and final_one_shot["mean_max_miss_db"] < best_pass_miss
                ):
                    best_pass = float(final_one_shot["pass_rate"])
                    best_pass_miss = float(final_one_shot["mean_max_miss_db"])
                    best_epochs["pass"] = last_epoch
                    self._save_checkpoint(
                        self.best_pass_path,
                        last_epoch,
                        {
                            "val_loss": float(history[-1]["validation"]["loss"]),
                            "val_mae": float(history[-1]["validation"]["mae"]),
                            "epoch": last_epoch,
                            "criterion": "one_shot_pass_rate",
                            "pass_rate": best_pass,
                            "mean_max_miss_db": best_pass_miss,
                        },
                    )

        train_seconds = float(sum(record["train"]["seconds"] for record in history))
        total_seconds = float(time.perf_counter() - fit_start)
        summary = {
            "output_dir": str(self.output_dir),
            "best_checkpoint": str(self.best_path),
            "best_loss_checkpoint": str(self.best_loss_path),
            "best_mae_checkpoint": str(self.best_mae_path),
            "best_pass_checkpoint": str(self.best_pass_path),
            "last_checkpoint": str(self.last_path),
            "best_val_loss": float(best_loss),
            "best_val_mae": float(best_mae),
            "best_one_shot_pass_rate": float(max(best_pass, 0.0)),
            "best_one_shot_mean_miss_db": float(best_pass_miss),
            "best_epochs": best_epochs,
            "epochs": len(history),
            "max_epochs": total_epochs,
            "trainable_parameters": count_parameters(self.model),
            "device": str(self.device),
            "amp": bool(self.use_amp),
            "amp_dtype": str(self.amp_dtype),
            "optimizer": str(self.cfg.get("optimizer", "adamw")),
            "scheduler": str(self.cfg.get("scheduler", "cosine")),
            "learning_rate": float(self.cfg.get("learning_rate", 3e-4)),
            "weight_decay": float(self.cfg.get("weight_decay", 1e-4)),
            "batch_size": int(self.cfg.get("batch_size", 32)),
            "grad_clip": float(self.cfg.get("grad_clip", 1.0)),
            "loss": str(self.cfg.get("loss", "huber")),
            "seed": int(self.cfg.get("seed", 2026)),
            "early_stopping_metric": early_metric,
            "early_stopping_patience": patience,
            "checkpoint_selection": [
                "val_loss",
                "val_mae",
                "one_shot_pass_rate",
                "last",
            ],
            "simulator_calls_during_training": int(self.simulator_calls),
            "one_shot_states": int(len(self.one_shot_seeds or [])),
            "one_shot_every": int(self.cfg.get("one_shot_every", 0) or 0),
            "peak_gpu_memory_mb": (
                float(torch.cuda.max_memory_allocated() / 2**20)
                if self.device.type == "cuda"
                else None
            ),
            "train_seconds": train_seconds,
            "total_seconds": total_seconds,
            "samples_per_second": float(
                sum(record["train"]["states"] for record in history)
                / max(train_seconds, 1e-9)
            ),
        }
        save_json(self.output_dir / "summary.json", summary)
        return summary


def load_checkpoint_into(model: torch.nn.Module, path: str) -> dict:
    checkpoint = torch.load(resolve(path), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    return {
        "epoch": int(checkpoint.get("epoch", -1)),
        "val_loss": float(checkpoint.get("val_loss", float("nan"))),
    }
