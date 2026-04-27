"""슬라이딩 윈도우 학습/평가로 시간에 따른 성능 변화를 측정한다.

본 데모 데이터셋은 negative sampling 구간이 2023-01 ~ 2024-12 이므로
해당 구간만 의미 있는 평가 결과가 된다. 데이터가 더 많은 환경에서는
``--start`` 를 앞당기면 더 긴 시계열 진단이 가능하다.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from marine_accident_risk.config import Config
from marine_accident_risk.eval.drift import plot_drift, sliding_evaluation

logger = logging.getLogger("drift")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--eval-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--start", default=None,
                        help="평가 데이터 시작 시점 (예: 2023-01-01).")
    parser.add_argument("--out-dir", default="reports/drift")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    df = pd.read_parquet(cfg.get("paths", "dataset"))
    if args.start:
        df = df[pd.to_datetime(df["ts"]) >= pd.Timestamp(args.start)].copy()

    feat_cols = [c for c in df.columns if c.startswith("wx_") or c in (
        "lat_center", "lon_center", "hour", "dow", "month", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "gid_acc_log1p", "hour_dow_acc_log1p", "cell_area_deg2",
    )]
    model_cfg = cfg.get("model")
    table = sliding_evaluation(
        df, feat_cols, "y", "ts",
        params=model_cfg["params"],
        window_train_months=args.train_months,
        eval_months=args.eval_months,
        step_months=args.step_months,
        n_estimators=int(model_cfg["params"].get("n_estimators", 400)),
        early_stopping_rounds=int(model_cfg["cv"]["early_stopping_rounds"]),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "drift_table.csv", index=False)
    plot_drift(table, out_dir / "drift_curve.png")

    valid = table.dropna(subset=["auc", "pr_auc"])
    if not valid.empty:
        summary = {
            "n_windows": int(len(table)),
            "n_valid_windows": int(len(valid)),
            "auc_mean": float(valid["auc"].mean()),
            "auc_std": float(valid["auc"].std()),
            "auc_min": float(valid["auc"].min()),
            "auc_max": float(valid["auc"].max()),
            "pr_auc_mean": float(valid["pr_auc"].mean()),
            "pr_auc_std": float(valid["pr_auc"].std()),
        }
    else:
        summary = {"n_windows": int(len(table)), "n_valid_windows": 0}

    import json
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote -> %s | summary=%s", out_dir, summary)


if __name__ == "__main__":
    main()
