"""시간 슬라이딩 학습/평가 — 컨셉드리프트 감지용.

학습 기간(window_train_months) 만큼 학습 → 다음 eval_months 만큼 평가를
한 칸씩 밀면서 반복한다. 평가 구간의 AUC / PR-AUC / positive rate / N 을
기록해 시간에 따른 성능 저하 또는 사고 패턴 변화를 진단한다.

본 데모 데이터셋은 negative sampling 이 ``time_window`` 구간(2023-2024) 안에서만
이루어지므로 해당 구간만 의미 있는 평가가 된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class DriftWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    n_train: int
    n_eval: int
    eval_pos_rate: float
    auc: float | None
    pr_auc: float | None


def _train_eval_window(df: pd.DataFrame, feature_cols: list[str],
                       label_col: str, params: dict,
                       train_mask: np.ndarray, eval_mask: np.ndarray,
                       n_estimators: int, early_stopping_rounds: int) -> tuple[float | None, float | None]:
    if eval_mask.sum() == 0 or train_mask.sum() == 0:
        return None, None
    y_tr = df.loc[train_mask, label_col].astype(int).to_numpy()
    y_va = df.loc[eval_mask, label_col].astype(int).to_numpy()
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
        return None, None
    dtr = lgb.Dataset(df.loc[train_mask, feature_cols], label=y_tr)
    dva = lgb.Dataset(df.loc[eval_mask, feature_cols], label=y_va, reference=dtr)
    booster = lgb.train(
        {**params}, dtr, num_boost_round=n_estimators,
        valid_sets=[dva], valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    pred = booster.predict(df.loc[eval_mask, feature_cols],
                           num_iteration=booster.best_iteration)
    return float(roc_auc_score(y_va, pred)), float(average_precision_score(y_va, pred))


def sliding_evaluation(df: pd.DataFrame, feature_cols: list[str],
                       label_col: str, ts_col: str, params: dict,
                       window_train_months: int = 6,
                       eval_months: int = 1,
                       step_months: int = 1,
                       n_estimators: int = 400,
                       early_stopping_rounds: int = 30) -> pd.DataFrame:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)

    period = df[ts_col].dt.to_period("M")
    months = sorted(period.unique())
    if not months:
        return pd.DataFrame()

    rows: list[DriftWindow] = []
    i = 0
    while i + window_train_months + eval_months <= len(months):
        tr_months = months[i: i + window_train_months]
        ev_months = months[i + window_train_months: i + window_train_months + eval_months]
        tr_mask = period.isin(tr_months).to_numpy()
        ev_mask = period.isin(ev_months).to_numpy()
        auc, pr_auc = _train_eval_window(
            df, feature_cols, label_col, params,
            tr_mask, ev_mask,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
        )
        rows.append(DriftWindow(
            train_start=tr_months[0].to_timestamp(),
            train_end=tr_months[-1].to_timestamp(how="end"),
            eval_start=ev_months[0].to_timestamp(),
            eval_end=ev_months[-1].to_timestamp(how="end"),
            n_train=int(tr_mask.sum()),
            n_eval=int(ev_mask.sum()),
            eval_pos_rate=float(df.loc[ev_mask, label_col].mean()) if ev_mask.sum() else float("nan"),
            auc=auc,
            pr_auc=pr_auc,
        ))
        i += step_months

    return pd.DataFrame([w.__dict__ for w in rows])


def plot_drift(table: pd.DataFrame, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = table.dropna(subset=["auc", "pr_auc"]).copy()
    if plot_df.empty:
        return out_path
    plot_df["eval_label"] = plot_df["eval_start"].dt.strftime("%Y-%m")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(plot_df["eval_label"], plot_df["auc"], marker="o", label="AUC")
    ax.plot(plot_df["eval_label"], plot_df["pr_auc"], marker="s", label="PR-AUC")
    ax.set_ylabel("score")
    ax.set_xlabel("evaluation month")
    ax.set_title("Sliding-window performance over time")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
