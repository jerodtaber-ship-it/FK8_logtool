from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Segment:
    kind: str
    start_idx: int
    end_idx: int

def _rpm_slope(time_s: np.ndarray, rpm: np.ndarray) -> np.ndarray:
    dt = np.diff(time_s, prepend=time_s[0])
    dr = np.diff(rpm, prepend=rpm[0])
    slope = np.divide(dr, np.where(dt == 0, np.nan, dt))
    return np.nan_to_num(slope, nan=0.0)

def detect_wot_pulls(df: pd.DataFrame,
                     app_thresh: float = 95.0,
                     min_rpm_slope: float = 50.0,
                     min_duration_s: float = 1.5) -> list[Segment]:
    """
    Detect WOT pulls using pedal position and rising RPM.
    - app_pct > app_thresh
    - rpm slope > min_rpm_slope (rpm/sec)
    """
    if not {"time_s","rpm","app_pct"}.issubset(df.columns):
        return []

    t = df["time_s"].to_numpy()
    rpm = df["rpm"].to_numpy()
    app = df["app_pct"].to_numpy()

    slope = _rpm_slope(t, rpm)
    is_wot = (app >= app_thresh) & (slope >= min_rpm_slope)

    segs: list[Segment] = []
    in_seg = False
    s = 0
    for i, flag in enumerate(is_wot):
        if flag and not in_seg:
            in_seg = True
            s = i
        if in_seg and (not flag or i == len(is_wot)-1):
            e = i if not flag else i
            in_seg = False
            dur = t[e] - t[s]
            if dur >= min_duration_s and (rpm[e] - rpm[s]) > 500:
                segs.append(Segment(kind="WOT_PULL", start_idx=s, end_idx=e))
    return segs

def detect_steady_cruise(df: pd.DataFrame,
                         app_max: float = 20.0,
                         rpm_slope_max: float = 40.0,
                         min_duration_s: float = 10.0,
                         throttle_min: float = 5.0,
                         throttle_max: float = 40.0) -> list[Segment]:
    """
    Detect steady cruise windows for trim/leak diagnosis.
    Heuristic:
      - low pedal
      - near-constant RPM (low rpm slope)
      - moderate throttle (avoid decel fuel cut & idle)
    """
    req = {"time_s","rpm","app_pct","throttle_pct"}
    if not req.issubset(df.columns):
        return []

    t = df["time_s"].to_numpy()
    rpm = df["rpm"].to_numpy()
    app = df["app_pct"].to_numpy()
    thr = df["throttle_pct"].to_numpy()
    slope = np.abs(_rpm_slope(t, rpm))

    steady = (app <= app_max) & (slope <= rpm_slope_max) & (thr >= throttle_min) & (thr <= throttle_max)

    segs: list[Segment] = []
    in_seg = False
    s = 0
    for i, flag in enumerate(steady):
        if flag and not in_seg:
            in_seg = True
            s = i
        if in_seg and (not flag or i == len(steady)-1):
            e = i if not flag else i
            in_seg = False
            dur = t[e] - t[s]
            if dur >= min_duration_s:
                segs.append(Segment(kind="CRUISE", start_idx=s, end_idx=e))
    return segs

def slice_segment(df: pd.DataFrame, seg: Segment) -> pd.DataFrame:
    return df.iloc[seg.start_idx:seg.end_idx+1].copy()
