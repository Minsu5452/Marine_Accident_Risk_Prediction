from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BBox:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def contains(self, lat: float, lon: float) -> bool:
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max


@dataclass
class Config:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            return cls(raw=yaml.safe_load(f))

    @property
    def bbox(self) -> BBox:
        b = self.raw["bbox"]
        return BBox(b["lat_min"], b["lat_max"], b["lon_min"], b["lon_max"])

    def get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur
