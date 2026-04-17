from __future__ import annotations

"""
Shared helpers for deep sequence models.

This engine imports ``build_sequences`` from ``dataset_builder`` and exposes
shared training utilities consumed by ``tcn.py``, ``tft.py``, and ``nbeat.py``.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from helper_classes.dataset_builder import build_sequences
from helper_classes.dataset_builder import inject_val_labels


def seed_all(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch."""
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reduce_weighted_loss(loss_vec: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    """Reduce per-sample losses with positive finite weights."""
    if loss_vec.ndim != 1:
        loss_vec = loss_vec.view(-1)

    sw = sample_weight.view(-1).to(loss_vec.device).float()
    keep = torch.isfinite(sw) & (sw > 0) & torch.isfinite(loss_vec)
    if torch.count_nonzero(keep) == 0:
        raise ValueError("No eligible elements for weighted loss reduction.")

    l = loss_vec[keep]
    w = sw[keep]
    return (l * w).sum() / (w.sum() + 1e-12)


@dataclass(frozen=True)
class SeqMeta:
    metro: np.ndarray
    year_snap: np.ndarray
    iso_week: np.ndarray
    week_start: np.ndarray
    target_week_start: np.ndarray
    y_true: np.ndarray
    weight: np.ndarray


def _rebuild_seq_weights_np(
    w: np.ndarray,
    *,
    mode: str = "capped",
    cap_q: float = 0.95,
) -> np.ndarray:
    """Apply capped or unweighted sequence weights and normalize if used."""
    ww = np.asarray(w, dtype=float).copy()
    keep = np.isfinite(ww) & (ww > 0)

    if int(keep.sum()) == 0:
        ww[~keep] = 0.0
        return ww.astype(np.float32)

    mode = str(mode).lower().strip()

    if mode == "capped":
        if not (0.0 < float(cap_q) <= 1.0):
            raise ValueError(f"cap_q must be in (0, 1]; got {cap_q}")
        cap = float(np.quantile(ww[keep], float(cap_q)))
        ww[keep] = np.minimum(ww[keep], cap)

        mu = float(np.mean(ww[keep]))
        if np.isfinite(mu) and mu > 0:
            ww[keep] = ww[keep] / mu
        else:
            raise ValueError("Bad mean after sequence weight capping.")

    elif mode == "unweighted":
        ww[keep] = 1.0

    else:
        raise ValueError(
            f"WEIGHT_MODE={mode!r}. Expected 'capped' or 'unweighted'."
        )

    ww[~keep] = 0.0
    return ww.astype(np.float32)


def _apply_seq_weight(
    loader: DataLoader,
    *,
    mode: str = "capped",
    cap_q: float = 0.95,
    shuffle: bool = True,
) -> DataLoader:
    """Return a loader with the weighting scheme."""
    tensors = tuple(loader.dataset.tensors)

    if len(tensors) == 3:
        X, y, w = tensors
        w2_np = _rebuild_seq_weights_np(w.detach().cpu().numpy(), mode=mode, cap_q=float(cap_q))
        w2 = torch.as_tensor(w2_np, dtype=w.dtype, device=w.device)
        ds2 = TensorDataset(X, y, w2)

    elif len(tensors) == 4:
        X, m, y, w = tensors
        w2_np = _rebuild_seq_weights_np(w.detach().cpu().numpy(), mode=mode, cap_q=float(cap_q))
        w2 = torch.as_tensor(w2_np, dtype=w.dtype, device=w.device)
        ds2 = TensorDataset(X, m, y, w2)

    return DataLoader(
        ds2,
        batch_size=int(loader.batch_size),
        shuffle=bool(shuffle),
        drop_last=False,
    )


def make_seq_loader(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    min_observed_weeks: int,
    batch_size: int,
    shuffle: bool,
    return_meta: bool = True,
    verbose: bool = True,
    mode: str,
    metro_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[DataLoader, Optional[SeqMeta]]:
    """Build a sequence DataLoader from the weekly panel."""
    if verbose:
        n = int(len(df))
        wv = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).to_numpy()
        n_posw = int(np.sum(wv > 0))
        metros = int(df["metro"].nunique()) if "metro" in df.columns else -1
        print(
            f"[model_builder] make_seq_loader({mode}): "
            f"rows={n:,} pos_weight_rows={n_posw:,} metros={metros:,}"
        )

    X, y, w, metro, ys, wk, ws = build_sequences(
        df,
        feature_cols=list(feature_cols),
        min_observed_weeks=int(min_observed_weeks),
    )

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    w_t = torch.from_numpy(w).float()

    if metro_to_idx is not None:
        unk = int(len(metro_to_idx))
        metro_idx = np.asarray(
            [metro_to_idx.get(str(m), unk) for m in np.asarray(metro, dtype=object)],
            dtype=np.int64,
        )
        m_t = torch.from_numpy(metro_idx).long()
        ds = TensorDataset(X_t, m_t, y_t, w_t)
    else:
        ds = TensorDataset(X_t, y_t, w_t)

    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=bool(shuffle), drop_last=False)

    meta = None
    if return_meta:
        key = pd.DataFrame(
            {
                "metro": np.asarray(metro, dtype=str),
                "week_start": pd.to_datetime(np.asarray(ws), errors="raise"),
            }
        )
        lookup = df.loc[:, ["metro", "week_start", "target_week_start"]].copy()
        lookup["metro"] = lookup["metro"].astype(str)
        lookup["week_start"] = pd.to_datetime(lookup["week_start"], errors="raise")
        lookup["target_week_start"] = pd.to_datetime(lookup["target_week_start"], errors="raise")

        merged = key.merge(lookup, on=["metro", "week_start"], how="left", validate="many_to_one")
        if merged["target_week_start"].isna().any():
            raise ValueError("Missing target_week_start in sequence metadata.")

        meta = SeqMeta(
            metro=np.asarray(metro, dtype=object),
            year_snap=np.asarray(ys, dtype=int),
            iso_week=np.asarray(wk, dtype=int),
            week_start=np.asarray(ws),
            target_week_start=merged["target_week_start"].to_numpy(dtype="datetime64[ns]"),
            y_true=np.asarray(y, dtype=int),
            weight=np.asarray(w, dtype=float),
        )

    return loader, meta


@dataclass(frozen=True)
class OptimConfig:
    """Optimizer settings."""

    lr: float
    weight_decay: float
    grad_clip_norm: float
    warmup_frac: float = 0.10


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> optim.Optimizer:
    """Build AdamW with no decay on bias and norm params."""
    if not isinstance(cfg, OptimConfig):
        raise TypeError(f"cfg must be OptimConfig; got {type(cfg).__name__}")

    lr = float(cfg.lr)
    wd = float(cfg.weight_decay)

    if not np.isfinite(lr) or lr <= 0:
        raise ValueError(f"Bad lr: {cfg.lr}")
    if not np.isfinite(wd) or wd < 0:
        raise ValueError(f"Bad weight_decay: {cfg.weight_decay}")

    norm_param_ids: set[int] = set()
    norm_types = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    for _, module in model.named_modules():
        if isinstance(module, norm_types):
            for p in module.parameters(recurse=False):
                norm_param_ids.add(id(p))

    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_bias = name.endswith(".bias") or name.endswith("bias")
        is_norm_param = id(p) in norm_param_ids

        if is_bias or is_norm_param:
            no_decay.append(p)
        else:
            decay.append(p)

    if (len(decay) + len(no_decay)) == 0:
        raise ValueError("No trainable parameters found.")

    param_groups = []
    if decay:
        param_groups.append({"params": decay, "weight_decay": wd})
    if no_decay:
        param_groups.append({"params": no_decay, "weight_decay": 0.0})

    return optim.AdamW(param_groups, lr=lr)


def build_cosine_warmup_scheduler(
    optimizer: optim.Optimizer,
    *,
    total_steps: int,
    cfg: OptimConfig,
) -> LambdaLR:
    """Build a cosine schedule with optional warmup."""
    if not isinstance(cfg, OptimConfig):
        raise TypeError(f"cfg must be OptimConfig; got {type(cfg).__name__}")

    total_steps = int(total_steps)
    if total_steps <= 0:
        return LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    warmup_frac = float(cfg.warmup_frac)
    if not np.isfinite(warmup_frac) or not (0.0 <= warmup_frac <= 1.0):
        raise ValueError(f"Bad warmup_frac: {warmup_frac}")

    warmup_steps = int(max(0, round(warmup_frac * total_steps)))
    warmup_steps = int(min(warmup_steps, total_steps))

    def lr_lambda(step: int) -> float:
        step = int(step)
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        denom = float(max(1, total_steps - warmup_steps))
        t = float(step - warmup_steps) / denom
        t = min(max(t, 0.0), 1.0)
        return 0.5 * (1.0 + float(np.cos(np.pi * t)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def clip_gradients(model: nn.Module, cfg: OptimConfig) -> float:
    """Apply global-norm clipping."""
    if not isinstance(cfg, OptimConfig):
        raise TypeError(f"cfg must be OptimConfig; got {type(cfg).__name__}")

    clip = float(cfg.grad_clip_norm)
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError(f"Bad grad_clip_norm: {cfg.grad_clip_norm}")

    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip))


def pack_oof_from_meta(
    meta: SeqMeta,
    prob: Sequence[float],
    *,
    fold: int,
    T0p9: float,
    tau: float,
) -> pd.DataFrame:
    """Pack probabilities and sequence metadata into the OOF schema."""
    p = np.asarray(prob, dtype=float).ravel()
    y = np.asarray(meta.y_true, dtype=float).ravel()
    w = np.asarray(meta.weight, dtype=float).ravel()

    n = int(y.shape[0])
    if p.shape[0] != n or w.shape[0] != n:
        raise ValueError(f"pack_oof_from_meta: length mismatch p={p.shape[0]} y={n} w={w.shape[0]}")
    if not np.all(np.isfinite(p)):
        raise ValueError("pack_oof_from_meta: non-finite prob")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("pack_oof_from_meta: prob outside [0,1]")
    if not np.all(np.isfinite(w)):
        raise ValueError("pack_oof_from_meta: non-finite weight")
    if np.any(w < 0):
        raise ValueError("pack_oof_from_meta: negative weight")
    if not np.all(np.isfinite(y)):
        raise ValueError("pack_oof_from_meta: non-finite y_true")
    if np.any(~np.isin(y, [0.0, 1.0])):
        raise ValueError("pack_oof_from_meta: y_true must be binary")

    return pd.DataFrame(
        {
            "metro": np.asarray(meta.metro),
            "week_start": np.asarray(meta.week_start),
            "target_week_start": np.asarray(meta.target_week_start),
            "year_snap": np.asarray(meta.year_snap, dtype=int),
            "iso_week": np.asarray(meta.iso_week, dtype=int),
            "fold": int(fold),
            "y_true": y.astype(int),
            "weight": w.astype(float),
            "prob": p.astype(float),
            "tau": float(tau),
            "T0p9": float(T0p9),
            "target_iso_year": np.asarray(meta.year_snap, dtype=int),
            "target_iso_week": np.asarray(meta.iso_week, dtype=int),
        }
    )


def make_train_seq_loader(
    train_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    min_observed_weeks: int,
    batch_size: int,
    shuffle: bool = True,
    return_meta: bool = True,
    verbose: bool = True,
    metro_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[DataLoader, Optional[SeqMeta]]:
    """Build the training loader."""
    return make_seq_loader(
        train_df,
        feature_cols=feature_cols,
        min_observed_weeks=int(min_observed_weeks),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        return_meta=bool(return_meta),
        verbose=bool(verbose),
        mode="train",
        metro_to_idx=metro_to_idx,
    )


def make_val_seq_loader(
    val_ctx_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    min_observed_weeks: int,
    batch_size: int,
    shuffle: bool = False,
    return_meta: bool = True,
    verbose: bool = True,
    metro_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[DataLoader, Optional[SeqMeta]]:
    """Build the validation loader from context plus labels."""

    ctx = inject_val_labels(val_ctx_df, val_df)
    return make_seq_loader(
        ctx,
        feature_cols=feature_cols,
        min_observed_weeks=int(min_observed_weeks),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        return_meta=bool(return_meta),
        verbose=bool(verbose),
        mode="val",
        metro_to_idx=metro_to_idx,
    )


def train_fixed_epochs_binary(
    model: nn.Module,
    loader: DataLoader,
    *,
    epochs: int,
    opt_cfg: OptimConfig,
    device: torch.device,
) -> None:
    """Train a deep binary classifier for fixed epochs."""
    epochs = int(epochs)
    if epochs <= 0:
        raise ValueError(f"epochs must be > 0; got {epochs}")

    total_steps = int(max(1, epochs * len(loader)))
    opt = build_optimizer(model, opt_cfg)
    sched = build_cosine_warmup_scheduler(opt, total_steps=total_steps, cfg=opt_cfg)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    model.train()
    for _ in range(epochs):
        for batch in loader:
            if len(batch) != 4:
                raise RuntimeError("Loader must provide metro_idx for deep models.")

            xb, mb, yb, wb = batch
            xb = xb.to(device)
            mb = mb.to(device).long().view(-1)
            yb = yb.to(device).float().view(-1)
            wb = wb.to(device).float().view(-1)

            opt.zero_grad(set_to_none=True)
            logits = model(xb, mb).view(-1)
            per = loss_fn(logits, yb)
            loss = reduce_weighted_loss(per, wb)
            loss.backward()
            clip_gradients(model, opt_cfg)
            opt.step()
            sched.step()


def predict_probs_from_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> np.ndarray:
    """Return predicted probabilities in loader order."""
    model.eval()
    out: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                xb, _, _ = batch
                mb = None
            elif len(batch) == 4:
                xb, mb, _, _ = batch
            else:
                raise ValueError(f"Unexpected batch arity {len(batch)}")

            xb = xb.to(device)
            if mb is None:
                logits = model(xb).view(-1)
            else:
                logits = model(xb, mb.to(device).long()).view(-1)

            prob = torch.sigmoid(logits).detach().cpu().numpy().astype(float)
            out.append(prob)

    if not out:
        return np.zeros((0,), dtype=float)
    return np.concatenate(out, axis=0).astype(float)

def _apply_weekly_weight_scheme(
    weekly: pd.DataFrame,
    *,
    mode: str,
    cap_q: float,
) -> pd.DataFrame:
    """Weights for capped or unweighted analysis."""
    out = weekly.copy()

    n = pd.to_numeric(out["n_reports"], errors="raise").to_numpy(dtype=float)
    w = np.zeros_like(n, dtype=float)

    obs = np.isfinite(n) & (n > 0)
    mode = str(mode).lower().strip()

    if mode == "capped":
        cap_q = float(cap_q)
        if not (0.0 < cap_q <= 1.0):
            raise ValueError(f"cap_q must be in (0, 1]; got {cap_q}")

        if np.any(obs):
            cap = float(np.quantile(n[obs], cap_q))
            n_cap = np.minimum(n[obs], cap)
            mu = float(np.mean(n_cap))
            if not np.isfinite(mu) or mu <= 0:
                raise ValueError("Bad mean after weight capping.")
            w[obs] = n_cap / mu

    elif mode == "unweighted":
        w[obs] = 1.0

    else:
        raise ValueError(
            f"WEIGHT_MODE={mode!r}. Expected 'capped' or 'unweighted'."
        )

    out["weight"] = w.astype(float)
    return out


def run_deep_outer_cv(
    weekly: pd.DataFrame,
    *,
    val_years: Sequence[int],
    policy: Any,
    feature_cols: Sequence[str],
    min_observed_weeks: int,
    weight_mode: str = "capped",
    weight_cap_q: float = 0.95,
    batch_size: int,
    epochs: int,
    opt_cfg: OptimConfig,
    device: torch.device,
    model_factory: Callable[..., nn.Module],
    ece_bins: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run outer rolling-origin CV for a deep model."""
    from helper_classes import dataset_builder as du
    from helper_classes.model_contract import (
        apply_impute_standardize,
        concat_and_validate_oof,
        evaluate_from_oof,
        feature_columns_present,
        fit_impute_standardize,
        fold_diagnostics_row,
        print_fold_diagnostics,
        require_model_columns,
        weight_rule,
        fold_prevalence_thresholds,
    )

    weekly = _apply_weekly_weight_scheme(
        weekly,
        mode=str(weight_mode),
        cap_q=float(weight_cap_q),
    )

    require_model_columns(weekly, require_label=False)
    feature_columns_present(weekly, feature_cols)
    weight_rule(weekly)

    def _meta_to_diag_df(meta: SeqMeta) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "metro": pd.Series(np.asarray(meta.metro, dtype=object)).astype(str),
                "week_start": pd.to_datetime(np.asarray(meta.week_start), errors="raise"),
                "target_week_start": pd.to_datetime(np.asarray(meta.target_week_start), errors="raise"),
                "flu_label": pd.to_numeric(np.asarray(meta.y_true), errors="coerce"),
                "weight": pd.to_numeric(np.asarray(meta.weight), errors="coerce"),
            }
        )

    fold_rows: List[Dict[str, Any]] = []
    train_oof_list: List[pd.DataFrame] = []
    val_oof_list: List[pd.DataFrame] = []

    for fold_id, vy in enumerate(list(val_years)):
        fold_id = int(fold_id)

        train_df, val_df, val_ctx_df, T0p9, _ = du.make_fold_frames(
            weekly,
            val_year=int(vy),
            policy=policy,
        )

        med, mu, sd = fit_impute_standardize(train_df, feature_cols, weight_col="weight")
        train_df_p = apply_impute_standardize(train_df, feature_cols, med, mu, sd)
        val_ctx_df_p = apply_impute_standardize(val_ctx_df, feature_cols, med, mu, sd)
        val_df_p = apply_impute_standardize(val_df, feature_cols, med, mu, sd)

        metros = sorted(pd.Series(train_df_p["metro"].astype(str).unique()).tolist())
        metro_to_idx = {m: j for j, m in enumerate(metros)}
        n_metros = int(len(metro_to_idx) + 1)

        train_loader, train_meta = make_train_seq_loader(
            train_df_p,
            feature_cols=feature_cols,
            min_observed_weeks=int(min_observed_weeks),
            batch_size=int(batch_size),
            shuffle=True,
            return_meta=True,
            verbose=bool(verbose),
            metro_to_idx=metro_to_idx,
        )

        val_loader, val_meta = make_val_seq_loader(
            val_ctx_df_p,
            val_df_p,
            feature_cols=feature_cols,
            min_observed_weeks=int(min_observed_weeks),
            batch_size=int(batch_size),
            shuffle=False,
            return_meta=True,
            verbose=bool(verbose),
            metro_to_idx=metro_to_idx,
        )

        train_eval_loader = DataLoader(
            train_loader.dataset,
            batch_size=int(train_loader.batch_size),
            shuffle=False,
            drop_last=False,
        )

        model = model_factory(int(len(feature_cols)), n_metros=int(n_metros)).to(device)

        train_fixed_epochs_binary(
            model,
            train_loader,
            epochs=int(epochs),
            opt_cfg=opt_cfg,
            device=device,
        )
        p_train = predict_probs_from_loader(model, train_eval_loader, device=device)
        p_val = predict_probs_from_loader(model, val_loader, device=device)

        if train_meta is None or val_meta is None:
            raise RuntimeError("Expected sequence metadata.")

        y_meta = np.asarray(train_meta.y_true, dtype=float)
        w_meta = np.asarray(train_meta.weight, dtype=float)
        keep = np.isfinite(w_meta) & (w_meta > 0) & np.isfinite(y_meta) & np.isin(y_meta, [0.0, 1.0])
        if not np.any(keep):
            raise ValueError(f"Fold {fold_id}: no eligible train rows for tau.")
        tau_eff = float(np.average(y_meta[keep], weights=w_meta[keep]))

        train_oof = pack_oof_from_meta(
            train_meta,
            p_train,
            fold=fold_id,
            T0p9=float(T0p9),
            tau=float(tau_eff),
        )
        val_oof = pack_oof_from_meta(
            val_meta,
            p_val,
            fold=fold_id,
            T0p9=float(T0p9),
            tau=float(tau_eff),
        )

        train_oof_list.append(train_oof)
        val_oof_list.append(val_oof)

        fold_rows.append(
            fold_diagnostics_row(
                fold=fold_id,
                val_year=int(vy),
                train_df=_meta_to_diag_df(train_meta),
                val_df=_meta_to_diag_df(val_meta),
                T0p9=float(T0p9),
                tau=float(tau_eff),
                thr_B=None,
            )
        )

    train_oof_all = concat_and_validate_oof(train_oof_list, name="train_oof_all")
    val_oof_all = concat_and_validate_oof(val_oof_list, name="val_oof_all")

    threshold_by_fold = fold_prevalence_thresholds(train_oof_all)
    for r in fold_rows:
        r["thr_B"] = float(threshold_by_fold[int(r["fold"])])

    fold_diag_df = print_fold_diagnostics(fold_rows)
    metrics = evaluate_from_oof(
        val_oof_all,
        threshold_by_fold=threshold_by_fold,
        ece_bins=int(ece_bins),
    )

    return {
        "train_oof_all": train_oof_all,
        "val_oof_all": val_oof_all,
        "threshold_by_fold": threshold_by_fold,
        "fold_diag_df": fold_diag_df,
        "metrics": metrics,
    }


__all__ = [
    "seed_all",
    "reduce_weighted_loss",
    "SeqMeta",
    "make_seq_loader",
    "OptimConfig",
    "build_optimizer",
    "build_cosine_warmup_scheduler",
    "clip_gradients",
    "pack_oof_from_meta",
    "make_train_seq_loader",
    "make_val_seq_loader",
    "train_fixed_epochs_binary",
    "predict_probs_from_loader",
    "run_deep_outer_cv",
]
