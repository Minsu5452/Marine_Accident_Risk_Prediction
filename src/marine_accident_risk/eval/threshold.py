"""Multi-threshold evaluation on LightGBM OOF predictions.

LightGBM 분류기의 OOF 예측을 받아 임계값(0.1~0.9)별 confusion-matrix 기반
지표(precision, recall, F1, specificity)와 세 가지 기준의 최적 임계값
(F1 최대 / Youden's J 최대 / cost-weighted) 을 계산한다.

운영 관점에서는 사고를 놓치는 비용(FN)이 잘못된 경보(FP)보다 크다고 가정,
cost-weighted 기준은 ``cost_fn / cost_fp`` 비율을 외부에서 받는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ThresholdReport:
    table: pd.DataFrame
    best_f1: dict
    best_youden: dict
    best_cost: dict
    auc: float
    pr_auc: float


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                         threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def evaluate_grid(y_true: np.ndarray, y_prob: np.ndarray,
                  thresholds: np.ndarray | None = None,
                  cost_fn: float = 5.0,
                  cost_fp: float = 1.0) -> ThresholdReport:
    """``thresholds`` 그리드에 대해 metrics_at_threshold 계산 + best 임계값 산출."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.1, 0.91, 0.1), 2)
    rows = [metrics_at_threshold(y_true, y_prob, t) for t in thresholds]
    table = pd.DataFrame(rows)
    table["youden_j"] = table["recall"] + table["specificity"] - 1.0
    table["expected_cost"] = cost_fn * table["fn"] + cost_fp * table["fp"]

    best_f1 = table.loc[table["f1"].idxmax()].to_dict()
    best_youden = table.loc[table["youden_j"].idxmax()].to_dict()
    best_cost = table.loc[table["expected_cost"].idxmin()].to_dict()

    return ThresholdReport(
        table=table,
        best_f1=best_f1,
        best_youden=best_youden,
        best_cost=best_cost,
        auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
    )


def plot_curves(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path,
                report: ThresholdReport) -> Path:
    """ROC + PR curve 저장. matplotlib backend는 Agg 사용."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    p, r, _ = precision_recall_curve(y_true, y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(fpr, tpr, label=f"AUC = {report.auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC curve (OOF)")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(r, p, label=f"PR-AUC = {report.pr_auc:.3f}")
    base = float((y_true == 1).mean())
    axes[1].axhline(base, color="grey", linestyle="--", linewidth=0.8,
                    label=f"baseline = {base:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curve (OOF)")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def write_markdown(report: ThresholdReport, out_path: Path,
                   cost_fn: float, cost_fp: float) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = report.table.copy()
    df_disp = df.assign(
        precision=df["precision"].round(3),
        recall=df["recall"].round(3),
        specificity=df["specificity"].round(3),
        f1=df["f1"].round(3),
        youden_j=df["youden_j"].round(3),
        expected_cost=df["expected_cost"].round(1),
    )

    def _fmt(d: dict) -> str:
        return (f"threshold={d['threshold']:.2f}, precision={d['precision']:.3f}, "
                f"recall={d['recall']:.3f}, F1={d['f1']:.3f}")

    lines = [
        "# Multi-threshold evaluation (LightGBM OOF)",
        "",
        f"- OOF AUC: **{report.auc:.4f}**",
        f"- OOF PR-AUC: **{report.pr_auc:.4f}**",
        f"- Cost weights: FN={cost_fn}, FP={cost_fp}",
        "",
        "## Threshold sweep",
        "",
        df_disp.to_markdown(index=False),
        "",
        "## Best operating points",
        "",
        f"- Best F1: {_fmt(report.best_f1)}",
        f"- Best Youden's J: {_fmt(report.best_youden)}",
        f"- Cost-weighted minimum (FN cost > FP cost): {_fmt(report.best_cost)}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
