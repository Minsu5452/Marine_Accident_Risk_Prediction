"""level4.csv 격자 로더 + 격자/위경도 매핑."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BBox

# level4.csv 컬럼 명에 lat/lon 이 거꾸로 들어 있어 의미 기준으로 정규화한다.
#   og_lon_min/max 의 실제 값 범위(28~44)는 위도 → lat
#   og_lat_min/max 의 실제 값 범위(120~136)는 경도 → lon
GRID_COL_RENAME = {
    "og_lon_min": "lat_min",
    "og_lon_max": "lat_max",
    "og_lat_min": "lon_min",
    "og_lat_max": "lon_max",
}

GRID_CELL_DEG = 0.025


def load_grid(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df = df.rename(columns=GRID_COL_RENAME)
    df["lat_center"] = (df["lat_min"] + df["lat_max"]) / 2
    df["lon_center"] = (df["lon_min"] + df["lon_max"]) / 2
    return df[["gid", "og_id", "lat_min", "lat_max", "lon_min", "lon_max",
               "lat_center", "lon_center"]]


def filter_grid(grid: pd.DataFrame, bbox: BBox) -> pd.DataFrame:
    mask = (
        (grid["lat_center"] >= bbox.lat_min)
        & (grid["lat_center"] <= bbox.lat_max)
        & (grid["lon_center"] >= bbox.lon_min)
        & (grid["lon_center"] <= bbox.lon_max)
    )
    return grid.loc[mask].reset_index(drop=True)


def assign_grid_id(grid: pd.DataFrame, lat: float, lon: float) -> int | None:
    """단일 위경도 → gid. demo용 단순 lookup; 실프로젝트는 spatial index 권장."""
    row = grid[
        (grid["lat_min"] <= lat) & (lat < grid["lat_max"])
        & (grid["lon_min"] <= lon) & (lon < grid["lon_max"])
    ]
    if row.empty:
        return None
    return int(row.iloc[0]["gid"])


def assign_grid_ids(grid: pd.DataFrame, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """벡터화된 격자 매핑. lat/lon 모두 셀 경계에 quantize 후 dict lookup."""
    lat_origin = float(grid["lat_min"].min())
    lon_origin = float(grid["lon_min"].min())

    lat_idx = np.floor((lats - lat_origin) / GRID_CELL_DEG).astype(int)
    lon_idx = np.floor((lons - lon_origin) / GRID_CELL_DEG).astype(int)

    g_lat_idx = np.floor((grid["lat_min"].to_numpy() - lat_origin) / GRID_CELL_DEG).astype(int)
    g_lon_idx = np.floor((grid["lon_min"].to_numpy() - lon_origin) / GRID_CELL_DEG).astype(int)
    lookup = {(int(la), int(lo)): int(gid)
              for la, lo, gid in zip(g_lat_idx, g_lon_idx, grid["gid"].to_numpy())}

    out = np.full(len(lats), -1, dtype=np.int64)
    for i, (la, lo) in enumerate(zip(lat_idx, lon_idx)):
        gid = lookup.get((int(la), int(lo)), -1)
        out[i] = gid
    return out
