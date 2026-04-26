"""공공데이터포털 (data.go.kr / 국립해양조사원) 클라이언트.

현재 활성화된 채널: surveySeafog (해무관측소).
나머지 5개 endpoint(Wind/AirPress/AirTemp/WaterTemp/TideLevel)는 동일 패턴이므로
활성화 확인 후 path만 swap 하면 된다.
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

DGK_BASE = "https://apis.data.go.kr/1192136"

ENDPOINTS = {
    "seafog":     "/surveySeafog/GetSurveySeafogApiService",
    "wind":       "/surveyWind/GetSurveyWindApiService",
    "airpress":   "/surveyAirPress/GetSurveyAirPressApiService",
    "airtemp":    "/surveyAirTemp/GetSurveyAirTempApiService",
    "watertemp":  "/surveyWaterTemp/GetSurveyWaterTempApiService",
    "tidelevel":  "/surveyTideLevel/GetSurveyTideLevelApiService",
}


class DataGoKrClient:
    def __init__(self, service_key: str | None = None, timeout: int = 30) -> None:
        self.service_key = service_key or os.environ.get("DATA_GO_KR_SERVICE_KEY")
        if not self.service_key:
            raise RuntimeError("DATA_GO_KR_SERVICE_KEY 환경변수 또는 service_key 인자가 필요합니다")
        self.timeout = timeout
        self.session = requests.Session()

    def fetch(self, kind: str, obs_code: str, req_date: date | datetime | str,
              minute_step: int = 60, page_no: int = 1, num_of_rows: int = 300) -> dict:
        if kind not in ENDPOINTS:
            raise ValueError(f"unknown kind: {kind}")
        path = ENDPOINTS[kind]
        if isinstance(req_date, (date, datetime)):
            ds = req_date.strftime("%Y%m%d")
        else:
            ds = str(req_date).replace("-", "")
        params = {
            "serviceKey": self.service_key,
            "type": "json",
            "obsCode": obs_code,
            "reqDate": ds,
            "pageNo": page_no,
            "numOfRows": num_of_rows,
        }
        if kind != "seafog":
            params["min"] = minute_step
        r = self.session.get(f"{DGK_BASE}{path}", params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def fetch_seafog_range(self, obs_codes: Iterable[str], start: date, end: date,
                           sleep: float = 0.3) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for code in obs_codes:
            cur = start
            while cur <= end:
                try:
                    payload = self.fetch("seafog", code, cur)
                except Exception as e:  # noqa: BLE001
                    logger.warning("seafog fetch fail (%s, %s): %s", code, cur, e)
                    time.sleep(sleep)
                    cur = (pd.Timestamp(cur) + pd.Timedelta(days=1)).date()
                    continue
                df = _parse_seafog(payload, code)
                if not df.empty:
                    frames.append(df)
                time.sleep(sleep)
                cur = (pd.Timestamp(cur) + pd.Timedelta(days=1)).date()
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


def _parse_seafog(payload: dict, obs_code: str) -> pd.DataFrame:
    header = payload.get("header") or {}
    if header.get("resultCode") != "00":
        return pd.DataFrame()
    items = ((payload.get("body") or {}).get("items") or {}).get("item") or []
    if not isinstance(items, list):
        items = [items]
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(items)
    df["obs_code"] = obs_code
    df["ts"] = pd.to_datetime(df["obsrvnDt"], errors="coerce")
    for c in ("lot", "lat", "rmyWspd", "amonAvgTp", "amonAvgHum",
              "amonAvgAtmpr", "amonAvgWtem", "dtvsbM20kLen", "dtvsbV20kLen"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
