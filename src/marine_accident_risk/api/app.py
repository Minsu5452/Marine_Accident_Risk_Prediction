"""FastAPI 추론 API.

GET /healthz
GET /grid              — bbox 안의 격자 메타
POST /predict          — body: {gid, ts, optional weather override}
POST /predict/grid     — body: {ts}; bbox 안 전체 격자 한 번에 추론
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import Config
from ..data.grid import filter_grid, load_grid
from ..features.build import _add_time_features  # internal reuse
from ..models import lgbm
from ..models.shap_analyzer import explain, top_contributors

CONFIG_PATH = os.environ.get("MARINE_CONFIG", "configs/default.yaml")
MODEL_PATH = os.environ.get("MARINE_MODEL", "models/lgbm.txt")

app = FastAPI(title="Marine Accident Risk API", version="0.1.0")

_state: dict = {}


def _ensure_loaded() -> None:
    if "booster" in _state:
        return
    cfg = Config.load(CONFIG_PATH)
    grid = load_grid(cfg.get("paths", "grid_csv"))
    grid = filter_grid(grid, cfg.bbox)
    booster, feat_cols = lgbm.load(MODEL_PATH)

    # 학습 시점의 grid prior / hour-dow prior 를 dataset.parquet 에서 그대로 복원
    gid_priors: dict[int, float] = {}
    hour_dow_priors: dict[tuple[int, int], float] = {}
    try:
        ds = pd.read_parquet(cfg.get("paths", "dataset"))
        gid_priors = (ds.dropna(subset=["gid"]).drop_duplicates("gid")
                      .set_index("gid")["gid_acc_log1p"].astype(float).to_dict())
        hour_dow_priors = {
            (int(h), int(d)): float(v)
            for (h, d), v in ds.drop_duplicates(["hour", "dow"])
                               .set_index(["hour", "dow"])["hour_dow_acc_log1p"]
                               .astype(float).items()
        }
    except Exception:  # noqa: BLE001
        pass

    _state.update({
        "cfg": cfg, "grid": grid, "booster": booster, "feature_cols": feat_cols,
        "gid_priors": gid_priors, "hour_dow_priors": hour_dow_priors,
    })


class PredictRequest(BaseModel):
    gid: int = Field(..., description="격자 id (level4.csv 의 gid)")
    ts: datetime = Field(..., description="대상 시각 (1시간 단위로 floor)")
    wind_speed: float | None = None
    air_temperature: float | None = None
    air_pressure: float | None = None
    humidity: float | None = None
    water_temper: float | None = None
    horizon_visibl: float | None = None


class PredictResponse(BaseModel):
    gid: int
    ts: datetime
    probability: float
    top_factors: list[tuple[str, float]]


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/grid")
def grid_meta() -> dict:
    _ensure_loaded()
    g = _state["grid"]
    return {
        "count": len(g),
        "bbox": _state["cfg"].raw["bbox"],
        "head": g.head(50).to_dict(orient="records"),
    }


def _row_for(gid: int, ts: datetime, overrides: dict[str, float | None]) -> pd.DataFrame:
    g = _state["grid"]
    cell = g[g["gid"] == gid]
    if cell.empty:
        raise HTTPException(status_code=404, detail=f"gid {gid} not in active grid")
    cell = cell.iloc[0]
    ts_floor = pd.Timestamp(ts).floor("h")
    row = {
        "gid": gid,
        "ts": ts_floor,
        "lat_center": float(cell["lat_center"]),
        "lon_center": float(cell["lon_center"]),
        "cell_area_deg2": (0.025 ** 2),
    }
    df = pd.DataFrame([row])
    df = pd.concat([df, _add_time_features(df["ts"]).reset_index(drop=True)], axis=1)
    for k in ("wind_speed", "air_temperature", "air_pressure",
              "humidity", "water_temper", "horizon_visibl"):
        v = overrides.get(k)
        df[f"wx_{k}"] = float(v) if v is not None else np.nan
    df["wx_distance_km"] = np.nan
    # 학습 시 priors 가 없으면 0 으로 fallback (LightGBM 은 NaN/0 모두 학습 분포 대비 안전)
    df["gid_acc_log1p"] = float(_state.get("gid_priors", {}).get(gid, 0.0))
    df["hour_dow_acc_log1p"] = float(_state.get("hour_dow_priors", {}).get(
        (int(df["hour"].iloc[0]), int(df["dow"].iloc[0])), 0.0))
    return df


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    _ensure_loaded()
    overrides = {
        "wind_speed": req.wind_speed,
        "air_temperature": req.air_temperature,
        "air_pressure": req.air_pressure,
        "humidity": req.humidity,
        "water_temper": req.water_temper,
        "horizon_visibl": req.horizon_visibl,
    }
    df = _row_for(req.gid, req.ts, overrides)
    cols = _state["feature_cols"]
    X = df[cols]
    proba = float(lgbm.predict_proba(_state["booster"], X)[0])
    sv = explain(_state["booster"], X).iloc[0]
    return PredictResponse(
        gid=req.gid,
        ts=df["ts"].iloc[0].to_pydatetime(),
        probability=proba,
        top_factors=top_contributors(sv, k=5),
    )


class GridPredictRequest(BaseModel):
    ts: datetime


@app.post("/predict/grid")
def predict_grid(req: GridPredictRequest) -> dict:
    _ensure_loaded()
    g = _state["grid"]
    ts_floor = pd.Timestamp(req.ts).floor("h")
    rows = []
    for _, cell in g.iterrows():
        rows.append({
            "gid": int(cell["gid"]),
            "ts": ts_floor,
            "lat_center": float(cell["lat_center"]),
            "lon_center": float(cell["lon_center"]),
            "cell_area_deg2": (0.025 ** 2),
        })
    df = pd.DataFrame(rows)
    df = pd.concat([df, _add_time_features(df["ts"]).reset_index(drop=True)], axis=1)
    for k in ("wind_speed", "air_temperature", "air_pressure",
              "humidity", "water_temper", "horizon_visibl"):
        df[f"wx_{k}"] = np.nan
    df["wx_distance_km"] = np.nan
    df["gid_acc_log1p"] = df["gid"].map(_state.get("gid_priors") or {}).fillna(0.0)
    h0, d0 = int(df["hour"].iloc[0]), int(df["dow"].iloc[0])
    df["hour_dow_acc_log1p"] = float((_state.get("hour_dow_priors") or {}).get((h0, d0), 0.0))
    cols = _state["feature_cols"]
    proba = lgbm.predict_proba(_state["booster"], df[cols])
    return {
        "ts": ts_floor.isoformat(),
        "predictions": [
            {"gid": int(g_), "lat": float(la), "lon": float(lo), "probability": float(p)}
            for g_, la, lo, p in zip(df["gid"], df["lat_center"], df["lon_center"], proba)
        ],
    }
