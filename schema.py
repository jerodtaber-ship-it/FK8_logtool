from __future__ import annotations
from dataclasses import dataclass
import re
import pandas as pd

# ---- Canonical column names used internally ----
CANON = {
    "time_s", "rpm", "gear", "app_pct", "throttle_pct",
    "afr_actual", "afr_cmd", "stft_pct", "ltft_pct",
    "frp_actual_psi", "frp_target_psi", "hpfp_spill_ff", "hpfp_spill_final", "fuel_status",
    "boost_actual_psi", "boost_target_psi", "map_psi", "baro_psi",
    "pre_throttle_p_act_psi", "pre_throttle_p_des_psi", "emp_psi",
    "iat_c", "mat_c", "battery_v",
    "wg_pos_act", "wg_pos_des",
    "turbo_pid_i", "turbo_pid_pd", "turbo_comp_protect_psi",
    "knock_ref_v", "knock_ratio_1", "knock_ratio_2",
    "ign_cyl1", "ign_cyl2", "ign_cyl3", "ign_cyl4",
    "kr_cyl1", "kr_cyl2", "kr_cyl3", "kr_cyl4",
}

# ---- FK8 Cobb AP header mapping (exact names from your header) ----
FK8_COBB_MAP = {
    "Time (sec)": "time_s",
    "Engine Speed (RPM)": "rpm",
    "Gear (-)": "gear",
    "Accelerator Pedal Position (%)": "app_pct",
    "ETC Angle Actual (%)": "throttle_pct",

    "AFR Actual (AFR)": "afr_actual",
    "AFR Commanded (Final) (AFR)": "afr_cmd",
    "Short Term Fuel Trim (%)": "stft_pct",
    "Long Term Fuel Trim (%)": "ltft_pct",

    "FRP Actual (psi)": "frp_actual_psi",
    "FRP Desired (psi)": "frp_target_psi",
    "HPFP Spill Valve Duty Cycle (Feed Forward) (kW)": "hpfp_spill_ff",
    "HPFP Spill Valve Duty Cycle (Final) (kW)": "hpfp_spill_final",
    "Fuel Status (-)": "fuel_status",

    "Boost Pressure (psi)": "boost_actual_psi",
    "Target Boost Pressure (psi)": "boost_target_psi",
    "MAP (psi)": "map_psi",
    "Barometric Pressure (psi)": "baro_psi",

    "Pressure Upstream Throttle Actual (psi)": "pre_throttle_p_act_psi",
    "Pressure Upstream Throttle Desired (psi)": "pre_throttle_p_des_psi",
    "Exhaust Manifold Pressure (psi)": "emp_psi",

    "Intake Air Temperature (C)": "iat_c",
    "Manifold Air Temperature (C)": "mat_c",
    "Battery Voltage (V)": "battery_v",

    "Wastegate Position Actual (-)": "wg_pos_act",
    "Wastegate Position Desired (-)": "wg_pos_des",

    "Turbo PID I-Term (-)": "turbo_pid_i",
    "Turbo PID PD-Term (-)": "turbo_pid_pd",
    "Turbo Maximum Boost - Component Protection (psi)": "turbo_comp_protect_psi",

    "Knock Level (Reference) (V)": "knock_ref_v",
    "Knock Ratio (Int/Ref) (-)": "knock_ratio_1",
    "Knock Ratio (Int/Ref) (2) (V)": "knock_ratio_2",

    "Ignition Timing Cyl1 (kW)": "ign_cyl1",
    "Ignition Timing Cyl2 (kW)": "ign_cyl2",
    "Ignition Timing Cyl3 (kW)": "ign_cyl3",
    "Ignition Timing Cyl4 (kW)": "ign_cyl4",

    "Knock Retard Cyl1 (kW)": "kr_cyl1",
    "Knock Retard Cyl2 (kW)": "kr_cyl2",
    "Knock Retard Cyl3 (kW)": "kr_cyl3",
    "Knock Retard Cyl4 (kW)": "kr_cyl4",
}

@dataclass(frozen=True)
class MappingResult:
    df: pd.DataFrame
    mapped: dict
    missing: list

def _normalize_header(s: str) -> str:
    # collapse whitespace and strip
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s

def load_log(path: str) -> pd.DataFrame:
    """Load CSV or XLSX."""
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")

def map_columns(df: pd.DataFrame, mapping: dict = FK8_COBB_MAP) -> MappingResult:
    """Map platform columns to canonical names. Keeps only mapped columns."""
    cols = {_normalize_header(c): c for c in df.columns}
    mapped = {}
    out = pd.DataFrame()

    for raw_name, canon in mapping.items():
        key = _normalize_header(raw_name)
        if key in cols:
            out[canon] = df[cols[key]]
            mapped[raw_name] = canon

    missing = [k for k in mapping.keys() if _normalize_header(k) not in cols]
    return MappingResult(df=out, mapped=mapped, missing=missing)

def basic_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns; drop rows with no time/rpm."""
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["time_s", "rpm"])
    out = out.sort_values("time_s").reset_index(drop=True)
    return out
