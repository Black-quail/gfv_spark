from __future__ import annotations

"""
TCN baseline for the weekly metro sequence task.
"""

import os
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from helper_classes import dataset_builder as du
from helper_classes import evaluation as ev
from helper_classes import model_contract as mc
from helper_classes import model_engine as me


CONFIG: Dict[str, object] = {
    "SEED": 30,
    # Model input
    "DATA_DIR": os.getenv("GFV_SPARK_DATA_DIR", os.path.join("path", "model_input")),
    "OOF_DIR": os.getenv("GFV_SPARK_OOF_DIR", os.path.join("path", "ensemble")),
    "VAL_YEARS": (2021, 2022, 2023, 2024),
    "REPORT_YEARS": (2022, 2023, 2024),
    "POST_PANDEMIC_START": 2022,
    "Q_EVENT": 0.90, # Default = 0.90, sensitivity = 0.95
    "WEIGHT_MODE": "capped",  # Default = "capped", sensitivity = unweighted
    "WEIGHT_CAP_Q": 0.95,  # Default=0.95, sensitivity=0.90, ignored if unweighted
    "ECE_BINS": 10,
    "MIN_OBS_WEEKS": 26,
    "MIN_METRO_WEEKS": 26,
    "D_MODEL": 64,
    "N_BLOCKS": 6,
    "KERNEL": 3,
    "DROPOUT": 0.10,
    "EMB_DIM": 32,
    "METRO_AFFINE_ALPHA": 0.20,
    "METRO_AFFINE_BETA": 0.20,
    "MODEL_TAG": "tcn",
    "BATCH_SIZE": 256,
    "EPOCHS": 40,
    "LR": 1e-3,
    "WEIGHT_DECAY": 1e-4,
    "GRAD_CLIP": 1.0,
    "WRITE_TRAIN_PARQUET": False,
    "WRITE_TABLES": True,
}


class _Chomp1d(nn.Module):
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = int(chomp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp <= 0:
            return x
        return x[..., :-self.chomp]


class MetroAffine(nn.Module):
    """Metro-specific logit adapter."""

    def __init__(self, n_metros: int, *, alpha: float = 0.20, beta: float = 0.20):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.s = nn.Embedding(int(n_metros), 1)
        self.b = nn.Embedding(int(n_metros), 1)

        nn.init.zeros_(self.s.weight)
        nn.init.zeros_(self.b.weight)

    def forward(self, logits: torch.Tensor, metro_idx: torch.Tensor) -> torch.Tensor:
        if metro_idx is None:
            raise ValueError("metro_idx is required.")

        m = metro_idx.long()
        s_raw = self.s(m).squeeze(-1)
        b_raw = self.b(m).squeeze(-1)

        scale = 1.0 + self.alpha * torch.tanh(s_raw)
        shift = self.beta * torch.tanh(b_raw)
        return logits * scale + shift


class _TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (int(kernel) - 1) * int(dilation)

        self.conv1 = weight_norm(
            nn.Conv1d(
                int(in_ch),
                int(out_ch),
                int(kernel),
                padding=int(pad),
                dilation=int(dilation),
            )
        )
        self.chomp1 = _Chomp1d(int(pad))
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(float(dropout))

        self.conv2 = weight_norm(
            nn.Conv1d(
                int(out_ch),
                int(out_ch),
                int(kernel),
                padding=int(pad),
                dilation=int(dilation),
            )
        )
        self.chomp2 = _Chomp1d(int(pad))
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(float(dropout))

        self.down = nn.Conv1d(int(in_ch), int(out_ch), 1) if int(in_ch) != int(out_ch) else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)


class TCNBinary(nn.Module):
    """TCN baseline over fixed histories."""

    def __init__(
        self,
        *,
        n_features: int,
        d_model: int = 64,
        n_blocks: int = 6,
        kernel: int = 3,
        dropout: float = 0.10,
        n_metros: int = 0,
        emb_dim: int = 32,
        metro_affine_alpha: float = 0.20,
        metro_affine_beta: float = 0.20,
    ):
        super().__init__()
        self.in_norm = nn.LayerNorm(int(n_features))

        layers: List[nn.Module] = []
        ch_in = int(n_features)
        for b in range(int(n_blocks)):
            ch_out = int(d_model)
            layers.append(
                _TemporalBlock(
                    ch_in,
                    ch_out,
                    kernel=int(kernel),
                    dilation=2 ** int(b),
                    dropout=float(dropout),
                )
            )
            ch_in = ch_out

        self.tcn = nn.Sequential(*layers)

        emb_dim = int(emb_dim)
        self.metro_emb = nn.Embedding(int(n_metros), emb_dim)
        self.head = nn.Linear(int(d_model) + emb_dim, 1)

        self.metro_affine = MetroAffine(
            int(n_metros),
            alpha=float(metro_affine_alpha),
            beta=float(metro_affine_beta),
        )

    def forward(self, x: torch.Tensor, metro_idx: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x (B, L, F); got {tuple(x.shape)}")

        x = self.in_norm(x)
        z = self.tcn(x.transpose(1, 2))
        h = z[:, :, -1]

        b = h.shape[0]
        m = metro_idx.long().view(-1)
        if m.numel() != b:
            raise ValueError("metro_idx shape mismatch.")

        e = self.metro_emb(m)
        h2 = torch.cat([h, e], dim=1)

        logits = self.head(h2).squeeze(-1)
        logits = self.metro_affine(logits, m)
        return logits


def run_all(*, write_oof: bool = True, verbose: bool = True) -> Dict[str, Dict[str, float]]:
    """Run outer rolling-origin CV for the TCN baseline."""
    me.seed_all(int(CONFIG["SEED"]))

    data_dir = str(CONFIG["DATA_DIR"])
    oof_dir = str(CONFIG["OOF_DIR"])
    os.makedirs(oof_dir, exist_ok=True)

    val_years = tuple(int(y) for y in CONFIG["VAL_YEARS"])
    report_years = tuple(int(y) for y in CONFIG.get("REPORT_YEARS", val_years))
    report_years_set = set(report_years)

    q_event = float(CONFIG["Q_EVENT"])
    post_start = int(CONFIG.get("POST_PANDEMIC_START", min(val_years)))
    ece_bins = int(CONFIG["ECE_BINS"])

    base_tag = str(CONFIG.get("MODEL_TAG", "tcn"))
    model_tag = f"{base_tag}_metroEA"
    model_name = "TCN_METROEMB_AFFINE"

    weekly = du.prepare_weekly_panel(data_dir, verbose=bool(verbose))

    min_metro_weeks = int(CONFIG.get("MIN_METRO_WEEKS", CONFIG.get("MIN_OBS_WEEKS", 0)))
    if min_metro_weeks > 0:
        weekly = du.retain_metros_min_weeks(
            weekly,
            min_weeks=min_metro_weeks,
            verbose=bool(verbose),
        )

    feature_cols = mc.pick_feature_columns()

    if verbose:
        ev.print_run_config(
            model_name=model_name,
            weekly=weekly,
            config=CONFIG,
            feature_cols=feature_cols,
        )

        if str(CONFIG.get("WEIGHT_MODE", "capped")).lower().strip()== "capped":
            print(f"weight_mode={str(CONFIG.get("WEIGHT_MODE", "capped")).lower().strip()} | weight_cap_q={float(CONFIG.get('WEIGHT_CAP_Q', 0.95))}")
        else:
            print(f"weight_mode={str(CONFIG.get("WEIGHT_MODE", "capped")).lower().strip()}")

        print("reporting_lag_weeks=0, fixed epochs")
        print(
            f"VAL_YEARS={val_years} | REPORT_YEARS={report_years} | "
            f"q_event={q_event} | post_start={post_start}"
        )

    policy = du.ThresholdPolicy(q_event=q_event, post_start_year=post_start)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt_cfg = me.OptimConfig(
        lr=float(CONFIG["LR"]),
        weight_decay=float(CONFIG["WEIGHT_DECAY"]),
        grad_clip_norm=float(CONFIG["GRAD_CLIP"]),
        warmup_frac=0.10,
    )

    def _model_factory(n_features: int, *, n_metros: Optional[int] = None) -> nn.Module:
        if n_metros is None:
            raise ValueError("n_metros is required.")

        return TCNBinary(
            n_features=int(n_features),
            d_model=int(CONFIG["D_MODEL"]),
            n_blocks=int(CONFIG["N_BLOCKS"]),
            kernel=int(CONFIG["KERNEL"]),
            dropout=float(CONFIG["DROPOUT"]),
            n_metros=int(n_metros),
            emb_dim=int(CONFIG["EMB_DIM"]),
            metro_affine_alpha=float(CONFIG["METRO_AFFINE_ALPHA"]),
            metro_affine_beta=float(CONFIG["METRO_AFFINE_BETA"]),
        )

    out = me.run_deep_outer_cv(
        weekly,
        val_years=val_years,
        policy=policy,
        feature_cols=feature_cols,
        min_observed_weeks=int(CONFIG["MIN_OBS_WEEKS"]),
        batch_size=int(CONFIG["BATCH_SIZE"]),
        epochs=int(CONFIG["EPOCHS"]),
        opt_cfg=opt_cfg,
        device=device,
        model_factory=_model_factory,
        ece_bins=ece_bins,
        verbose=bool(verbose),
        weight_mode=str(CONFIG.get("WEIGHT_MODE", "capped")),
        weight_cap_q=float(CONFIG.get("WEIGHT_CAP_Q", 0.95)),
    )

    train_oof_all = out["train_oof_all"]
    val_oof_all = out["val_oof_all"]
    threshold_by_fold = out["threshold_by_fold"]
    fold_diag_df = out["fold_diag_df"]

    if "year_snap" not in val_oof_all.columns:
        raise ValueError("val_oof_all missing year_snap; cannot filter REPORT_YEARS.")

    val_oof_report = val_oof_all.loc[val_oof_all["year_snap"].isin(report_years_set)].copy()
    if val_oof_report.empty:
        raise ValueError(
            f"[{model_tag}] Empty report cohort for REPORT_YEARS={sorted(report_years_set)}."
        )

    fold_diag_report = fold_diag_df.loc[
        pd.to_numeric(fold_diag_df["val_year"], errors="raise").isin(report_years_set)
    ].copy()

    report_folds = set(
        pd.to_numeric(fold_diag_report["fold"], errors="raise").astype(int).tolist()
    )
    threshold_by_fold_report = {
        int(f): float(v)
        for f, v in threshold_by_fold.items()
        if int(f) in report_folds
    }

    metrics = mc.evaluate_from_oof(
        val_oof_report,
        threshold_by_fold=threshold_by_fold,
        ece_bins=ece_bins,
    )

    if write_oof:
        # full OOF for ensemble training
        mc.write_validated_oof(
            val_oof_all,
            path=os.path.join(oof_dir, f"oof_{model_tag}.parquet"),
            name=f"oof_{model_tag}",
        )

        # report-only sidecar
        mc.write_validated_oof(
            val_oof_report,
            path=os.path.join(oof_dir, f"oof_{model_tag}_report.parquet"),
            name=f"oof_{model_tag}_report",
        )

        if bool(CONFIG.get("WRITE_TRAIN_PARQUET", False)):
            mc.write_validated_oof(
                train_oof_all,
                path=os.path.join(oof_dir, f"train_oof_{model_tag}.parquet"),
                name=f"train_oof_{model_tag}",
            )

        if bool(CONFIG.get("WRITE_TABLES", True)):
            fold_diag_report.to_csv(
                os.path.join(oof_dir, f"fold_diag_{model_tag}_report.csv"),
                index=False,
            )
            pd.DataFrame(
                [{"fold": int(k), "thr_B": float(v)} for k, v in sorted(threshold_by_fold_report.items())]
            ).to_csv(
                os.path.join(oof_dir, f"threshold_by_fold_{model_tag}_report.csv"),
                index=False,
            )

    if verbose:
        print(f"\n[ {model_tag} ] OOF metrics (REPORT_YEARS only)")
        print(ev.format_metrics_table([{"model": model_name, **metrics}], index_key="model"))

    return {model_tag: dict(metrics)}


if __name__ == "__main__":
    run_all(write_oof=True, verbose=True)
