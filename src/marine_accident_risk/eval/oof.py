"""StratifiedKFold OOF 예측 생성 유틸. lgbm.train_cv 의 가벼운 변형."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class OOFResult:
    y_true: np.ndarray
    y_prob: np.ndarray
    oof_auc: float
    oof_pr_auc: float
    fold_metrics: list[dict]
    ts: pd.Series | None = None


def compute_oof(df: pd.DataFrame, feature_cols: list[str], label_col: str,
                params: dict, n_splits: int = 5, n_estimators: int = 400,
                early_stopping_rounds: int = 30, seed: int = 42,
                ts_col: str | None = None) -> OOFResult:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = df[feature_cols].copy()
    y = df[label_col].astype(int).to_numpy()
    oof = np.zeros(len(df))
    fold_metrics = []

    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        dtr = lgb.Dataset(X.iloc[tr], label=y[tr])
        dva = lgb.Dataset(X.iloc[va], label=y[va], reference=dtr)
        booster = lgb.train(
            {**params},
            dtr,
            num_boost_round=n_estimators,
            valid_sets=[dtr, dva],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        pred = booster.predict(X.iloc[va], num_iteration=booster.best_iteration)
        oof[va] = pred
        fold_metrics.append({
            "fold": fold,
            "auc": float(roc_auc_score(y[va], pred)),
            "pr_auc": float(average_precision_score(y[va], pred)),
            "best_iter": int(booster.best_iteration or 0),
        })

    ts = df[ts_col] if ts_col and ts_col in df.columns else None
    return OOFResult(
        y_true=y,
        y_prob=oof,
        oof_auc=float(roc_auc_score(y, oof)),
        oof_pr_auc=float(average_precision_score(y, oof)),
        fold_metrics=fold_metrics,
        ts=ts,
    )
