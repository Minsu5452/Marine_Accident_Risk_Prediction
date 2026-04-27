"""negative_sampling.ratio 변경에 따른 OOF AUC / PR-AUC 비교.

ratio in {1, 3, 5, 10} 으로 dataset 을 다시 빌드해 5-fold OOF 학습을
수행하고 결과를 표로 정리한다. positive sample 수는 동일하므로 비교
가능한 ablation 이 된다.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import pandas as pd

from marine_accident_risk.config import Config
from marine_accident_risk.data.accidents import load_accidents
from marine_accident_risk.data.grid import load_grid
from marine_accident_risk.eval.oof import compute_oof
from marine_accident_risk.features.build import build_dataset

logger = logging.getLogger("ablation")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ratios", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--weather", default=None,
                        help="optional NMPNT weather parquet (없으면 wx_* = NaN)")
    parser.add_argument("--out-dir", default="reports/ablation")
    args = parser.parse_args()

    base_cfg = Config.load(args.config)
    accidents = load_accidents(base_cfg.get("paths", "accidents_xlsx"))
    grid = load_grid(base_cfg.get("paths", "grid_csv"))
    weather = pd.read_parquet(args.weather) if args.weather else pd.DataFrame()

    rows: list[dict] = []
    for ratio in args.ratios:
        cfg_raw = copy.deepcopy(base_cfg.raw)
        cfg_raw["negative_sampling"]["ratio"] = int(ratio)
        cfg = Config(raw=cfg_raw)
        bundle = build_dataset(cfg, accidents, grid, weather)
        n_pos = int((bundle.df["y"] == 1).sum())
        n_neg = int((bundle.df["y"] == 0).sum())
        logger.info("ratio=%d  n_pos=%d  n_neg=%d", ratio, n_pos, n_neg)

        model_cfg = cfg.get("model")
        oof = compute_oof(
            bundle.df, bundle.feature_cols, bundle.label_col,
            params=model_cfg["params"],
            n_splits=int(model_cfg["cv"]["n_splits"]),
            early_stopping_rounds=int(model_cfg["cv"]["early_stopping_rounds"]),
            n_estimators=int(model_cfg["params"].get("n_estimators", 400)),
        )
        rows.append({
            "ratio": ratio,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "pos_rate": n_pos / (n_pos + n_neg),
            "oof_auc": oof.oof_auc,
            "oof_pr_auc": oof.oof_pr_auc,
        })
        logger.info("  -> AUC=%.4f  PR-AUC=%.4f", oof.oof_auc, oof.oof_pr_auc)

    table = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "negative_ratio_table.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Negative-sampling ratio ablation",
        "",
        "negative_sampling.ratio 를 변경해 dataset 을 다시 빌드한 뒤 동일",
        "feature 셋과 LightGBM hyperparam 으로 5-fold OOF 성능을 비교했습니다.",
        "",
        table.assign(
            pos_rate=table["pos_rate"].round(3),
            oof_auc=table["oof_auc"].round(4),
            oof_pr_auc=table["oof_pr_auc"].round(4),
        ).to_markdown(index=False),
        "",
    ]
    (out_dir / "ablation_report.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("wrote -> %s", out_dir)


if __name__ == "__main__":
    main()
