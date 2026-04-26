"""Marine Accident Risk CLI.

Subcommands:
  weather-fetch   NMPNT 기상 한 기간 수집 → parquet
  build-dataset   사고 + 격자 + 기상 → 학습 데이터셋
  train           LightGBM CV 학습
  shap            모델 + 데이터셋 → SHAP parquet
  predict         단일 row 추론 (디버그용)
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .config import Config
from .data.accidents import load_accidents
from .data.grid import load_grid
from .data.nmpnt_client import NMPNTClient, parse_records, resample_hourly
from .features.build import build_dataset
from .models import lgbm
from .models.shap_analyzer import explain

logger = logging.getLogger("marine_risk")


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def cmd_weather_fetch(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    client = NMPNTClient()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for mmaf in args.mmaf:
        df = client.fetch_range_hourly(mmaf=mmaf, mmsi=args.mmsi.split(","),
                                       start=_parse_date(args.start),
                                       end=_parse_date(args.end))
        if not df.empty:
            frames.append(df)
    if not frames:
        logger.warning("no weather rows fetched")
        return
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(out_path)
    logger.info("wrote %d rows -> %s", len(out), out_path)


def cmd_build_dataset(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    accidents = load_accidents(cfg.get("paths", "accidents_xlsx"))
    grid = load_grid(cfg.get("paths", "grid_csv"))
    weather = pd.read_parquet(args.weather) if args.weather else pd.DataFrame()
    bundle = build_dataset(cfg, accidents, grid, weather)
    out = Path(cfg.get("paths", "dataset"))
    out.parent.mkdir(parents=True, exist_ok=True)
    bundle.df.to_parquet(out)
    logger.info("dataset rows=%d, features=%d -> %s",
                len(bundle.df), len(bundle.feature_cols), out)


def cmd_train(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    df = pd.read_parquet(cfg.get("paths", "dataset"))
    feat_cols = [c for c in df.columns if c.startswith("wx_") or c in (
        "lat_center", "lon_center", "hour", "dow", "month", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "gid_acc_log1p", "hour_dow_acc_log1p", "cell_area_deg2",
    )]
    model_cfg = cfg.get("model")
    res = lgbm.train_cv(
        df, feat_cols, "y",
        params=model_cfg["params"],
        n_splits=int(model_cfg["cv"]["n_splits"]),
        early_stopping_rounds=int(model_cfg["cv"]["early_stopping_rounds"]),
        n_estimators=int(model_cfg["params"].get("n_estimators", 400)),
    )
    lgbm.save(res, cfg.get("paths", "model"))
    logger.info("OOF AUC=%.4f PR_AUC=%.4f", res.oof_auc, res.oof_pr_auc)


def cmd_shap(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    df = pd.read_parquet(cfg.get("paths", "dataset"))
    booster, feat_cols = lgbm.load(cfg.get("paths", "model"))
    sv = explain(booster, df[feat_cols])
    out = Path(cfg.get("paths", "shap_cache"))
    out.parent.mkdir(parents=True, exist_ok=True)
    sv.to_parquet(out)
    logger.info("shap rows=%d -> %s", len(sv), out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="marine-risk")
    sub = p.add_subparsers(dest="cmd", required=True)
    p.add_argument("--config", default="configs/default.yaml")

    p_w = sub.add_parser("weather-fetch")
    p_w.add_argument("--start", required=True)
    p_w.add_argument("--end", required=True)
    p_w.add_argument("--mmaf", nargs="+", default=["101"])
    p_w.add_argument("--mmsi", required=True)
    p_w.add_argument("--out", default="data/processed/weather.parquet")
    p_w.set_defaults(func=cmd_weather_fetch)

    p_b = sub.add_parser("build-dataset")
    p_b.add_argument("--weather", default=None)
    p_b.set_defaults(func=cmd_build_dataset)

    p_t = sub.add_parser("train")
    p_t.set_defaults(func=cmd_train)

    p_s = sub.add_parser("shap")
    p_s.set_defaults(func=cmd_shap)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
