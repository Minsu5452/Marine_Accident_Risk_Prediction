"""LightGBM OOF 기반 multi-threshold 분석 실행 스크립트.

Usage:
  python scripts/run_threshold_analysis.py --config configs/default.yaml \
      --cost-fn 5 --cost-fp 1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from marine_accident_risk.config import Config
from marine_accident_risk.eval.oof import compute_oof
from marine_accident_risk.eval.threshold import (
    evaluate_grid,
    plot_curves,
    write_markdown,
)

logger = logging.getLogger("threshold")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cost-fn", type=float, default=5.0)
    parser.add_argument("--cost-fp", type=float, default=1.0)
    parser.add_argument("--out-dir", default="reports/threshold")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    df = pd.read_parquet(cfg.get("paths", "dataset"))
    feat_cols = [c for c in df.columns if c.startswith("wx_") or c in (
        "lat_center", "lon_center", "hour", "dow", "month", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "gid_acc_log1p", "hour_dow_acc_log1p", "cell_area_deg2",
    )]
    model_cfg = cfg.get("model")
    oof = compute_oof(
        df, feat_cols, "y",
        params=model_cfg["params"],
        n_splits=int(model_cfg["cv"]["n_splits"]),
        early_stopping_rounds=int(model_cfg["cv"]["early_stopping_rounds"]),
        n_estimators=int(model_cfg["params"].get("n_estimators", 400)),
    )
    logger.info("OOF AUC=%.4f PR_AUC=%.4f", oof.oof_auc, oof.oof_pr_auc)

    report = evaluate_grid(
        oof.y_true, oof.y_prob,
        cost_fn=args.cost_fn, cost_fp=args.cost_fp,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.table.to_csv(out_dir / "threshold_table.csv", index=False)
    plot_curves(oof.y_true, oof.y_prob, out_dir / "roc_pr_curves.png", report)
    write_markdown(report, out_dir / "threshold_report.md",
                   cost_fn=args.cost_fn, cost_fp=args.cost_fp)
    summary = {
        "oof_auc": report.auc,
        "oof_pr_auc": report.pr_auc,
        "best_f1": report.best_f1,
        "best_youden": report.best_youden,
        "best_cost": report.best_cost,
        "cost_fn": args.cost_fn,
        "cost_fp": args.cost_fp,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote -> %s", out_dir)


if __name__ == "__main__":
    main()
