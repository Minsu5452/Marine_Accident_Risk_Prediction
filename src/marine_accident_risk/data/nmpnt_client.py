"""국립해양측위정보원 OPEN API 클라이언트.

http://marineweather.nmpnt.go.kr:8001/openWeatherDate.do?...

분 단위 관측치를 받아서 1시간 단위로 리샘플링까지 처리한다.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime
from typing import Iterable

import pandas as pd
import requests

logger = logging.getLogger(__name__)

NMPNT_BASE = "http://marineweather.nmpnt.go.kr:8001"

NUMERIC_FIELDS = (
    "WIND_DIRECT", "WIND_SPEED",
    "SURFACE_CURR_DRC", "SURFACE_CURR_SPEED",
    "WAVE_DRC", "WAVE_HEIGTH",
    "AIR_TEMPERATURE", "HUMIDITY", "AIR_PRESSURE",
    "WATER_TEMPER", "SALINITY",
    "HORIZON_VISIBL",
    "TIDE_SPEED",
)


class NMPNTClient:
    def __init__(self, service_key: str | None = None, timeout: int = 30) -> None:
        self.service_key = service_key or os.environ.get("NMPNT_SERVICE_KEY")
        if not self.service_key:
            raise RuntimeError("NMPNT_SERVICE_KEY 환경변수 또는 service_key 인자가 필요합니다")
        self.timeout = timeout
        self.session = requests.Session()

    def fetch_now(self, mmaf: int | str, mmsi: Iterable[int | str], dataType: int = 1) -> dict:
        params = {
            "serviceKey": self.service_key,
            "resultType": "json",
            "mmaf": str(mmaf),
            "mmsi": ",".join(str(x) for x in mmsi),
            "dataType": dataType,
        }
        r = self.session.get(f"{NMPNT_BASE}/openWeatherNow.do", params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def fetch_date(self, mmaf: int | str, mmsi: Iterable[int | str],
                   target: date | datetime | str, dataType: int = 1) -> dict:
        if isinstance(target, (date, datetime)):
            ds = target.strftime("%Y%m%d")
        else:
            ds = str(target).replace("-", "")
        params = {
            "serviceKey": self.service_key,
            "resultType": "json",
            "date": ds,
            "mmaf": str(mmaf),
            "mmsi": ",".join(str(x) for x in mmsi),
            "dataType": dataType,
        }
        r = self.session.get(f"{NMPNT_BASE}/openWeatherDate.do", params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def fetch_range_hourly(self, mmaf: int | str, mmsi: Iterable[int | str],
                           start: date, end: date, sleep: float = 0.3) -> pd.DataFrame:
        """[start, end] 일 단위 순회 → 분 단위 관측치 → 1시간 평균/대표값 리샘플링."""
        frames: list[pd.DataFrame] = []
        cur = start
        while cur <= end:
            try:
                payload = self.fetch_date(mmaf, mmsi, cur)
            except Exception as e:  # noqa: BLE001
                logger.warning("NMPNT fetch fail (%s): %s", cur, e)
                cur = pd.Timestamp(cur) + pd.Timedelta(days=1)
                cur = cur.date()
                time.sleep(sleep)
                continue
            df = parse_records(payload)
            if not df.empty:
                frames.append(df)
            time.sleep(sleep)
            cur = pd.Timestamp(cur) + pd.Timedelta(days=1)
            cur = cur.date()
        if not frames:
            return pd.DataFrame()
        raw = pd.concat(frames, ignore_index=True)
        return resample_hourly(raw)


def parse_records(payload: dict) -> pd.DataFrame:
    result = payload.get("result", {})
    if result.get("status") != "OK":
        return pd.DataFrame()
    records = result.get("recordset") or []
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    df["ts"] = pd.to_datetime(df["DATETIME"], format="%Y%m%d%H%M%S", errors="coerce")
    for c in NUMERIC_FIELDS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "LATITUDE" in df.columns:
        df["lat"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    if "LONGITUDE" in df.columns:
        df["lon"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """station(MMSI_CODE) × 1시간 평균. 풍향 등 각도는 단순 평균(데모 단순화)."""
    keep = ["ts", "MMAF_CODE", "MMSI_CODE", "MMSI_NM", "lat", "lon", *NUMERIC_FIELDS]
    cols = [c for c in keep if c in df.columns]
    df = df[cols].copy()
    df["hour"] = df["ts"].dt.floor("h")
    agg: dict[str, str] = {c: "mean" for c in NUMERIC_FIELDS if c in df.columns}
    agg.update({"lat": "mean", "lon": "mean", "MMSI_NM": "first", "MMAF_CODE": "first"})
    out = df.groupby(["MMSI_CODE", "hour"], as_index=False).agg(agg)
    return out.rename(columns={"hour": "ts"})
