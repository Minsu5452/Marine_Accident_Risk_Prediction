"""격자×시간 학습 데이터셋 빌더.

positive  = 격자 안에서 해당 시간(±30분)에 발생한 사고
negative  = positive 격자/시간 외에서 무작위 샘플링 (config.negative_sampling.ratio)

각 row 의 features:
  - 시간: hour, dow, month, is_weekend, sin/cos cyclic
  - 격자: 위경도 중심, 격자 면적
  - 기상: 가장 가까운 NMPNT 측위정보원 station 의 1시간 평균 기상값
  - 선박용도: 격자×시간 ±lookback 안의 사고 비중 (전 기간 통계)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .. import data as _data_pkg  # noqa: F401  ensure subpackage importable
from ..config import Config
from ..data.grid import GRID_CELL_DEG, assign_grid_ids, filter_grid

logger = logging.getLogger(__name__)


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    feature_cols: list[str]
    label_col: str = "y"


def _add_time_features(ts: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=ts.index)
    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def _nearest_weather(events: pd.DataFrame, weather: pd.DataFrame,
                     max_distance_km: float, lookback_hours: int) -> pd.DataFrame:
    """events (lat/lon/ts) → 가장 가까운 station + lookback 시간 평균 기상."""
    if weather.empty:
        for c in ("WIND_SPEED", "AIR_TEMPERATURE", "AIR_PRESSURE", "HUMIDITY",
                  "WATER_TEMPER", "HORIZON_VISIBL"):
            events[f"wx_{c.lower()}"] = np.nan
        events["wx_distance_km"] = np.nan
        return events

    stations = (
        weather.dropna(subset=["lat", "lon"])
        .drop_duplicates("MMSI_CODE")[["MMSI_CODE", "lat", "lon"]]
        .reset_index(drop=True)
    )
    if stations.empty:
        return _nearest_weather(events, weather=pd.DataFrame(),
                                max_distance_km=max_distance_km,
                                lookback_hours=lookback_hours)
    tree = BallTree(np.radians(stations[["lat", "lon"]].values), metric="haversine")
    ev_xy = np.radians(events[["lat_center", "lon_center"]].rename(
        columns={"lat_center": "lat", "lon_center": "lon"}).values)
    dist_rad, idx = tree.query(ev_xy, k=1)
    dist_km = dist_rad[:, 0] * 6371.0
    nearest_station = stations.iloc[idx[:, 0]]["MMSI_CODE"].values
    events = events.copy()
    events["nearest_station"] = nearest_station
    events["wx_distance_km"] = dist_km

    feature_cols = ("WIND_SPEED", "AIR_TEMPERATURE", "AIR_PRESSURE", "HUMIDITY",
                    "WATER_TEMPER", "HORIZON_VISIBL")
    weather = weather.copy()
    weather["ts"] = pd.to_datetime(weather["ts"])
    wide = weather.pivot_table(index="ts", columns="MMSI_CODE",
                               values=list(feature_cols), aggfunc="mean")
    wide = wide.sort_index()

    join_keys = events[["ts", "nearest_station"]].copy()
    join_keys["ts"] = pd.to_datetime(join_keys["ts"]).dt.floor("h")
    out = events.copy()
    for c in feature_cols:
        col_name = f"wx_{c.lower()}"
        if c not in wide.columns.get_level_values(0):
            out[col_name] = np.nan
            continue
        col_block = wide[c]
        # lookback rolling mean per station
        rolled = col_block.rolling(window=lookback_hours, min_periods=1).mean()

        def _lookup(row, _rolled=rolled):
            try:
                return _rolled.loc[row["ts"], row["nearest_station"]]
            except KeyError:
                return np.nan
        out[col_name] = join_keys.apply(_lookup, axis=1)

    out.loc[out["wx_distance_km"] > max_distance_km,
            [f"wx_{c.lower()}" for c in feature_cols]] = np.nan
    return out


def _grid_priors(accidents: pd.DataFrame) -> pd.DataFrame:
    """격자(gid)별 historical 사고 건수 prior — positive/negative 모두에 동일 부여 가능."""
    g = accidents.groupby("gid").size().rename("gid_acc_count").reset_index()
    g["gid_acc_log1p"] = np.log1p(g["gid_acc_count"])
    return g[["gid", "gid_acc_log1p"]]


def _hour_dow_priors(accidents: pd.DataFrame) -> pd.DataFrame:
    g = (accidents.assign(hour=pd.to_datetime(accidents["ts"]).dt.hour,
                          dow=pd.to_datetime(accidents["ts"]).dt.dayofweek)
         .groupby(["hour", "dow"]).size()
         .rename("hour_dow_acc_count").reset_index())
    g["hour_dow_acc_log1p"] = np.log1p(g["hour_dow_acc_count"])
    return g[["hour", "dow", "hour_dow_acc_log1p"]]


def build_dataset(cfg: Config, accidents: pd.DataFrame, grid: pd.DataFrame,
                  weather: pd.DataFrame) -> DatasetBundle:
    bbox = cfg.bbox
    grid = filter_grid(grid, bbox).reset_index(drop=True)
    accidents = accidents.dropna(subset=["lat", "lon", "ts"]).copy()
    accidents = accidents[accidents["lat"].between(bbox.lat_min, bbox.lat_max)]
    accidents = accidents[accidents["lon"].between(bbox.lon_min, bbox.lon_max)]

    accidents["gid"] = assign_grid_ids(grid,
                                       accidents["lat"].to_numpy(),
                                       accidents["lon"].to_numpy())
    accidents = accidents[accidents["gid"] >= 0].copy()
    accidents["ts"] = pd.to_datetime(accidents["ts"]).dt.floor("h")

    pos = accidents.merge(grid[["gid", "lat_center", "lon_center"]], on="gid", how="left")
    pos["y"] = 1

    start = pd.Timestamp(cfg.get("time_window", "start"))
    end = pd.Timestamp(cfg.get("time_window", "end"))
    rng = pd.date_range(start, end, freq=cfg.get("time_window", "freq", default="1h"))
    ratio = int(cfg.get("negative_sampling", "ratio", default=5))
    seed = int(cfg.get("negative_sampling", "random_seed", default=42))
    rs = np.random.default_rng(seed)

    n_neg = ratio * len(pos)
    neg_gids = rs.choice(grid["gid"].to_numpy(), size=n_neg, replace=True)
    neg_ts = rs.choice(rng, size=n_neg, replace=True)
    neg = pd.DataFrame({"gid": neg_gids, "ts": neg_ts})
    neg = neg.merge(grid[["gid", "lat_center", "lon_center"]], on="gid", how="left")
    pos_keys = set(zip(pos["gid"], pos["ts"].astype("datetime64[ns]")))
    mask = [(g, t) not in pos_keys for g, t in zip(neg["gid"], neg["ts"].astype("datetime64[ns]"))]
    neg = neg.loc[mask].copy()
    neg["y"] = 0

    keep_cols = ["gid", "ts", "lat_center", "lon_center", "y"]
    df = pd.concat([pos[keep_cols], neg[keep_cols]], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    tf = _add_time_features(df["ts"]).reset_index(drop=True)
    df = pd.concat([df.reset_index(drop=True), tf], axis=1)

    df = _nearest_weather(df, weather,
                          max_distance_km=float(cfg.get("features", "weather_max_distance_km",
                                                        default=60)),
                          lookback_hours=int(cfg.get("features", "weather_lookback_hours",
                                                     default=3)))

    gp = _grid_priors(accidents)
    df = df.merge(gp, on="gid", how="left")
    df["gid_acc_log1p"] = df["gid_acc_log1p"].fillna(0)

    hp = _hour_dow_priors(accidents)
    df = df.merge(hp, on=["hour", "dow"], how="left")
    df["hour_dow_acc_log1p"] = df["hour_dow_acc_log1p"].fillna(0)

    df["cell_area_deg2"] = GRID_CELL_DEG ** 2

    feature_cols = [c for c in df.columns if c.startswith("wx_") or c in (
        "lat_center", "lon_center", "hour", "dow", "month", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "gid_acc_log1p", "hour_dow_acc_log1p", "cell_area_deg2",
    )]
    return DatasetBundle(df=df, feature_cols=feature_cols)
