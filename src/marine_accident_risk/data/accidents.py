"""해양사고 xlsx 로더 + 정규화."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import BBox

ACCIDENT_COL_RENAME = {
    "사건명": "case_name",
    "사고발생일시": "ts",
    "사고종류": "accident_type",
    "안전사고유형": "safety_type",
    "사망자(명)": "deaths",
    "실종자(명)": "missing",
    "사망·실종자(명)": "deaths_missing",
    "부상자(명)": "injuries",
    "해역": "sea_area",
    "선박용도(통계)": "ship_use_stat",
    "선박용도(대)": "ship_use_l",
    "선박용도(중)": "ship_use_m",
    "선박용도(소)": "ship_use_s",
    "선박톤수(톤)": "tonnage",
    "톤수(통계)": "tonnage_stat",
    "위도(º)": "lat",
    "경도(º)": "lon",
}


def load_accidents(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.rename(columns=ACCIDENT_COL_RENAME)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def filter_accidents(df: pd.DataFrame, bbox: BBox,
                     start: str | None = None, end: str | None = None) -> pd.DataFrame:
    mask = (
        df["lat"].between(bbox.lat_min, bbox.lat_max)
        & df["lon"].between(bbox.lon_min, bbox.lon_max)
    )
    if start is not None:
        mask &= df["ts"] >= pd.Timestamp(start)
    if end is not None:
        mask &= df["ts"] <= pd.Timestamp(end)
    return df.loc[mask].reset_index(drop=True)
