"""Streamlit 격자 위험도 대시보드.

- 시간 슬라이더로 1시간 단위 위험도 변화
- pydeck heatmap (격자 셀 polygon + 위험도 색상)
- 격자 클릭 → 그 격자의 SHAP top-5 기여도 표시 (간이: 위경도 기반 lookup)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from ..config import Config
from ..data.grid import filter_grid, load_grid
from ..features.build import _add_time_features
from ..models import lgbm
from ..models.shap_analyzer import explain, top_contributors


@st.cache_resource(show_spinner=False)
def _load_artifacts():
    cfg = Config.load(os.environ.get("MARINE_CONFIG", "configs/default.yaml"))
    grid = filter_grid(load_grid(cfg.get("paths", "grid_csv")), cfg.bbox)
    booster, feat_cols = lgbm.load(os.environ.get("MARINE_MODEL", "models/lgbm.txt"))
    return cfg, grid, booster, feat_cols


def _grid_df_for(grid: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    df = grid[["gid", "lat_center", "lon_center", "lat_min", "lat_max",
               "lon_min", "lon_max"]].copy()
    df["ts"] = ts
    df["ship_use_prior"] = 0.0
    df["cell_area_deg2"] = (0.025 ** 2)
    df = pd.concat([df, _add_time_features(df["ts"])], axis=1)
    for k in ("wind_speed", "air_temperature", "air_pressure",
              "humidity", "water_temper", "horizon_visibl"):
        df[f"wx_{k}"] = np.nan
    return df


def main() -> None:
    st.set_page_config(page_title="Marine Accident Risk Dashboard", layout="wide")
    st.title("Marine Accident Risk — 격자 × 시간")
    st.caption("⚠️ 본 대시보드는 회사 NDA 로 인해 본인이 별도로 재현한 데모입니다. "
               "데이터/모델은 공개 OPEN API + 공개 격자 + 공개 사고 통계만 사용합니다.")

    cfg, grid, booster, feat_cols = _load_artifacts()

    st.sidebar.header("설정")
    base_date = st.sidebar.date_input("날짜", value=datetime(2024, 6, 15).date())
    hour = st.sidebar.slider("시각 (KST)", 0, 23, value=14)
    show_top = st.sidebar.checkbox("상위 위험 격자만 표시", value=True)
    top_pct = st.sidebar.slider("상위 %", 1, 50, value=20) if show_top else 100

    ts = pd.Timestamp(base_date) + pd.Timedelta(hours=hour)
    df = _grid_df_for(grid, ts)
    df["proba"] = lgbm.predict_proba(booster, df[feat_cols])

    if show_top and len(df) > 0:
        thr = np.percentile(df["proba"], 100 - top_pct)
        view_df = df[df["proba"] >= thr].copy()
    else:
        view_df = df.copy()

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader(f"격자별 위험도 (ts={ts})")
        if view_df.empty:
            st.info("표시할 격자가 없습니다.")
        else:
            pmin, pmax = float(view_df["proba"].min()), float(view_df["proba"].max())
            view_df["norm"] = ((view_df["proba"] - pmin)
                               / max(pmax - pmin, 1e-9)).clip(0, 1)
            view_df["polygon"] = view_df.apply(
                lambda r: [
                    [r["lon_min"], r["lat_min"]],
                    [r["lon_max"], r["lat_min"]],
                    [r["lon_max"], r["lat_max"]],
                    [r["lon_min"], r["lat_max"]],
                ], axis=1)
            view_df["fill"] = view_df["norm"].apply(
                lambda v: [int(255 * v), int(60 * (1 - v)), int(60 * (1 - v)), 160])
            layer = pdk.Layer(
                "PolygonLayer",
                data=view_df,
                get_polygon="polygon",
                get_fill_color="fill",
                stroked=True,
                get_line_color=[40, 40, 40, 80],
                pickable=True,
                auto_highlight=True,
            )
            view = pdk.ViewState(
                latitude=float(view_df["lat_center"].mean()),
                longitude=float(view_df["lon_center"].mean()),
                zoom=9.5,
            )
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                                     tooltip={"text": "gid: {gid}\nprob: {proba}"}))

    with col2:
        st.subheader("격자 SHAP 분해")
        if df.empty:
            st.info("학습된 격자가 없습니다.")
        else:
            gid_pick = st.selectbox("격자 선택", view_df["gid"].astype(int).tolist()
                                    if not view_df.empty
                                    else df["gid"].astype(int).tolist())
            row = df[df["gid"] == gid_pick]
            sv = explain(booster, row[feat_cols]).iloc[0]
            top = top_contributors(sv, k=5)
            st.write(f"**예측 확률:** {float(row['proba'].iloc[0]):.4f}")
            st.write("**Top 기여 피처 (SHAP value)**")
            st.dataframe(pd.DataFrame(top, columns=["feature", "shap_value"]))


if __name__ == "__main__":
    main()
