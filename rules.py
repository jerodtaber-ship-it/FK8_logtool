from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Finding:
    code: str
    severity: str  # "info" | "warn" | "fail"
    title: str
    detail: str
    evidence: dict

def _pct(cond: np.ndarray) -> float:
    return float(np.mean(cond) * 100.0) if len(cond) else 0.0

def _safe_np(a):
    return np.asarray(a, dtype=float)

def infer_wg_polarity(seg: pd.DataFrame) -> str:
    """
    Best-effort guess:
    If wg_pos increases when boost error increases, likely 'higher = more open' (more opening -> less boost).
    """
    need = {"wg_pos_act","boost_actual_psi","boost_target_psi"}
    if not need.issubset(seg.columns):
        return "unknown"
    wg = _safe_np(seg["wg_pos_act"])
    err = _safe_np(seg["boost_target_psi"] - seg["boost_actual_psi"])
    if np.std(wg) < 1e-6 or np.std(err) < 1e-6:
        return "unknown"
    corr = float(np.corrcoef(wg, err)[0, 1])
    if corr > 0.2:
        return "higher=more_open"
    if corr < -0.2:
        return "higher=more_closed"
    return "unknown"

def emp_boost_ratio(seg: pd.DataFrame) -> tuple[float|None, float|None]:
    """
    Compute EMP:Boost ratio in two ways:
    - gauge ratio: EMP_g / Boost_g
    - absolute ratio: (EMP_g+baro)/(Boost_g+baro) when baro exists
    Returns (gauge_ratio_median, abs_ratio_median)
    """
    if not {"emp_psi","boost_actual_psi"}.issubset(seg.columns):
        return (None, None)
    emp = _safe_np(seg["emp_psi"])
    bst = _safe_np(seg["boost_actual_psi"])
    mask = bst > 2.0
    if not np.any(mask):
        return (None, None)
    gr = float(np.nanmedian(emp[mask] / bst[mask]))

    ar = None
    if "baro_psi" in seg.columns:
        baro = _safe_np(seg["baro_psi"])
        ar = float(np.nanmedian((emp[mask] + baro[mask]) / (bst[mask] + baro[mask])))
    return (gr, ar)

def emp_ratio_by_rpm(seg: pd.DataFrame, bins: list[int] | None = None) -> list[dict]:
    """
    Bin EMP:Boost ratio by RPM to show where the turbine starts choking.
    Returns list of dicts: [{"rpm_lo":..., "rpm_hi":..., "ratio_g_median":..., "ratio_abs_median":..., "n":...}, ...]
    """
    if bins is None:
        bins = [2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000]

    req = {"rpm","emp_psi","boost_actual_psi"}
    if not req.issubset(seg.columns):
        return []

    rpm = _safe_np(seg["rpm"])
    emp = _safe_np(seg["emp_psi"])
    bst = _safe_np(seg["boost_actual_psi"])
    baro = _safe_np(seg["baro_psi"]) if "baro_psi" in seg.columns else None

    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (rpm >= lo) & (rpm < hi) & (bst > 2.0)
        n = int(np.sum(m))
        if n < 8:
            continue
        g = float(np.nanmedian(emp[m] / bst[m]))
        a = None
        if baro is not None:
            a = float(np.nanmedian((emp[m] + baro[m]) / (bst[m] + baro[m])))
        out.append({"rpm_lo": int(lo), "rpm_hi": int(hi), "ratio_g_median": g, "ratio_abs_median": a, "n": n})
    return out

def run_wot_rules(seg: pd.DataFrame) -> list[Finding]:
    f: list[Finding] = []
    wgpol = infer_wg_polarity(seg)

    # ---- Throttle closure / torque limiting ----
    if {"app_pct","throttle_pct"}.issubset(seg.columns):
        app = _safe_np(seg["app_pct"])
        thr = _safe_np(seg["throttle_pct"])
        closure = (app >= 95) & (thr < 90)
        pct = _pct(closure)
        thr_min = float(np.nanmin(thr))
        if pct > 2:
            f.append(Finding(
                code="THROTTLE_CLOSURE",
                severity="warn",
                title="Throttle closure during WOT",
                detail=f"Throttle <90% for {pct:.1f}% of the pull while pedal >=95% (min throttle {thr_min:.1f}%). Often indicates torque/traction/thermal limiting.",
                evidence={"closure_pct": pct, "thr_min": thr_min}
            ))
        else:
            f.append(Finding(
                code="THROTTLE_OK",
                severity="info",
                title="Throttle stayed open",
                detail=f"Throttle <90% for {pct:.1f}% of the pull while pedal >=95% (min throttle {thr_min:.1f}%).",
                evidence={"closure_pct": pct, "thr_min": thr_min}
            ))

    # ---- Boost tracking ----
    underboost = False
    if {"boost_actual_psi","boost_target_psi","time_s"}.issubset(seg.columns):
        tgt = _safe_np(seg["boost_target_psi"])
        act = _safe_np(seg["boost_actual_psi"])
        err = tgt - act
        max_tgt = float(np.nanmax(tgt))
        max_act = float(np.nanmax(act))
        mean_err = float(np.nanmean(err))
        p90_err = float(np.nanpercentile(err, 90))
        p10_err = float(np.nanpercentile(err, 10))

        derr = np.diff(err, prepend=err[0])
        sign_changes = np.sum(np.sign(derr[1:]) != np.sign(derr[:-1]))
        osc_rate = float(sign_changes / max(1, len(err)-1))

        if p90_err > 2.0:
            underboost = True
            sev = "warn" if p90_err < 4.0 else "fail"
            f.append(Finding(
                code="UNDERBOOST",
                severity=sev,
                title="Underboost vs target",
                detail=(f"Peak target {max_tgt:.1f} psi, peak actual {max_act:.1f} psi. "
                        f"Boost error p90 {p90_err:.1f} psi (mean {mean_err:.1f}). "
                        f"WG polarity guess: {wgpol}. Oscillation rate {osc_rate:.2f}."),
                evidence={"max_target": max_tgt, "max_actual": max_act, "p90_err": p90_err,
                          "p10_err": p10_err, "mean_err": mean_err, "wg_polarity": wgpol,
                          "osc_rate": osc_rate}
            ))
        else:
            f.append(Finding(
                code="BOOST_OK",
                severity="info",
                title="Boost tracking looks good",
                detail=f"Peak target {max_tgt:.1f} psi, peak actual {max_act:.1f} psi. Boost error p90 {p90_err:.1f} psi.",
                evidence={"max_target": max_tgt, "max_actual": max_act, "p90_err": p90_err}
            ))

    # ---- Component protection limiting ----
    if {"turbo_comp_protect_psi","boost_target_psi","boost_actual_psi"}.issubset(seg.columns):
        protect = _safe_np(seg["turbo_comp_protect_psi"])
        tgt = _safe_np(seg["boost_target_psi"])
        act = _safe_np(seg["boost_actual_psi"])
        limiting = protect < (tgt - 0.8)
        pct_lim = _pct(limiting)
        if pct_lim > 5:
            hug = np.abs(act - protect) <= 0.7
            pct_hug = _pct(hug & limiting)
            f.append(Finding(
                code="COMP_PROTECT",
                severity="warn",
                title="Turbo component protection may be limiting boost",
                detail=f"Protection ceiling was >0.8 psi below target for {pct_lim:.1f}% of pull; actual boost was within 0.7 psi of ceiling for {pct_hug:.1f}% of those points.",
                evidence={"pct_limited": pct_lim, "pct_hugging_ceiling": pct_hug,
                          "protect_min": float(np.nanmin(protect)), "target_max": float(np.nanmax(tgt))}
            ))

    # ---- Wastegate position saturation hints ----
    if "wg_pos_act" in seg.columns:
        wg = _safe_np(seg["wg_pos_act"])
        wg_min = float(np.nanmin(wg))
        wg_max = float(np.nanmax(wg))
        p5 = float(np.nanpercentile(wg, 5))
        p95 = float(np.nanpercentile(wg, 95))
        f.append(Finding(
            code="WG_RANGE",
            severity="info",
            title="Wastegate position range",
            detail=f"WG pos p5={p5:.3f}, p95={p95:.3f} (min={wg_min:.3f}, max={wg_max:.3f}), polarity guess: {wgpol}.",
            evidence={"wg_p5": p5, "wg_p95": p95, "wg_min": wg_min, "wg_max": wg_max, "wg_polarity": wgpol}
        ))
        if underboost and wgpol == "higher=more_closed":
            closed_like = wg >= p95
            pct_closed = _pct(closed_like)
            if pct_closed > 30:
                f.append(Finding(
                    code="WG_SAT_CLOSED",
                    severity="warn",
                    title="WG appears near closed extreme during underboost",
                    detail=f"During underboost, WG spent {pct_closed:.1f}% of pull at/above p95 (closed-like by inferred polarity). Suggests mechanical airflow limit (leak, turbine choking, turbo efficiency) rather than control.",
                    evidence={"pct_closed_like": pct_closed, "wg_p95": p95}
                ))
        if underboost and wgpol == "higher=more_open":
            open_like = wg >= p95
            pct_open = _pct(open_like)
            if pct_open > 30:
                f.append(Finding(
                    code="WG_POLARITY_UNCERTAIN",
                    severity="info",
                    title="WG polarity may be inverted/uncertain",
                    detail=f"WG spent {pct_open:.1f}% near its high end while underboosting; confirm whether higher WG position means more open or more closed in your monitor definition.",
                    evidence={"pct_high_end": pct_open, "wg_p95": p95, "wg_polarity": wgpol}
                ))

    # ---- EMP : Boost ratio (overall + by RPM bins) ----
    gr, ar = emp_boost_ratio(seg)
    if gr is not None:
        sev = "info"
        if gr > 2.5:
            sev = "fail"
        elif gr > 2.2:
            sev = "warn"
        detail = f"Median EMP:Boost (gauge) ratio ≈ {gr:.2f}"
        if ar is not None:
            detail += f" (absolute ≈ {ar:.2f})"
        detail += ". Higher ratios often indicate turbine choking/restriction or pre-turb issues."
        f.append(Finding(
            code="EMP_RATIO",
            severity=sev,
            title="EMP to boost ratio",
            detail=detail,
            evidence={"emp_boost_gauge_ratio_median": gr, "emp_boost_abs_ratio_median": ar}
        ))

        binned = emp_ratio_by_rpm(seg)
        if binned:
            # Flag if ratio rises materially with RPM (simple: last bin - first bin)
            first = binned[0]["ratio_g_median"]
            last = binned[-1]["ratio_g_median"]
            delta = float(last - first)
            sev2 = "info"
            if last > 2.5 or delta > 0.6:
                sev2 = "warn"
            f.append(Finding(
                code="EMP_RATIO_RPM",
                severity=sev2,
                title="EMP:Boost by RPM (median per bin)",
                detail=f"EMP:Boost rises from ~{first:.2f} to ~{last:.2f} across bins (Δ≈{delta:.2f}). Useful to spot where turbine starts choking.",
                evidence={"bins": binned, "delta_first_last": delta}
            ))

    # ---- Fuel rail tracking + HPFP spill context ----
    if {"frp_actual_psi","frp_target_psi"}.issubset(seg.columns):
        tgt = _safe_np(seg["frp_target_psi"])
        act = _safe_np(seg["frp_actual_psi"])
        err = tgt - act
        max_drop = float(np.nanmax(err))
        min_act = float(np.nanmin(act))
        if max_drop > 300:
            sev = "warn" if max_drop < 600 else "fail"
            f.append(Finding(
                code="FRP_DROP",
                severity=sev,
                title="Fuel rail pressure not tracking target",
                detail=f"Max FRP shortfall {max_drop:.0f} psi (min actual {min_act:.0f} psi). Watch HPFP/LPFP capacity and commanded load.",
                evidence={"max_shortfall_psi": max_drop, "min_frp_actual_psi": min_act}
            ))
        else:
            f.append(Finding(
                code="FRP_OK",
                severity="info",
                title="Fuel rail pressure tracks well",
                detail=f"Max FRP shortfall {max_drop:.0f} psi (min actual {min_act:.0f} psi).",
                evidence={"max_shortfall_psi": max_drop, "min_frp_actual_psi": min_act}
            ))
        if {"hpfp_spill_final"}.issubset(seg.columns):
            spill = _safe_np(seg["hpfp_spill_final"])
            p95_spill = float(np.nanpercentile(spill, 95))
            hi = spill >= p95_spill
            pct_hi = _pct(hi)
            if max_drop > 300 and pct_hi > 25:
                f.append(Finding(
                    code="HPFP_AT_LIMIT",
                    severity="warn",
                    title="HPFP control appears near its high end during FRP shortfall",
                    detail=f"HPFP Spill Final spent {pct_hi:.1f}% of pull near/above its p95 while FRP shortfall exceeded 300 psi. Suggests pump/supply capacity or commanded target too aggressive.",
                    evidence={"pct_spill_high": pct_hi, "spill_p95": p95_spill, "max_frp_shortfall": max_drop}
                ))

    # ---- AFR tracking ----
    if {"afr_actual","afr_cmd"}.issubset(seg.columns):
        act = _safe_np(seg["afr_actual"])
        cmd = _safe_np(seg["afr_cmd"])
        err = act - cmd
        mean_abs = float(np.nanmean(np.abs(err)))
        lean_p90 = float(np.nanpercentile(err, 90))
        if mean_abs > 0.5 or lean_p90 > 0.8:
            f.append(Finding(
                code="AFR_MISS",
                severity="warn",
                title="AFR not tracking commanded",
                detail=f"Mean |AFR error| {mean_abs:.2f}, lean error p90 {lean_p90:.2f}. Check fuel pressure, injector calibration, and transient fueling.",
                evidence={"mean_abs_afr_error": mean_abs, "lean_error_p90": lean_p90}
            ))
        else:
            f.append(Finding(
                code="AFR_OK",
                severity="info",
                title="AFR tracking looks good",
                detail=f"Mean |AFR error| {mean_abs:.2f}, lean error p90 {lean_p90:.2f}.",
                evidence={"mean_abs_afr_error": mean_abs, "lean_error_p90": lean_p90}
            ))

    # ---- Knock ----
    kr_cols = [c for c in ["kr_cyl1","kr_cyl2","kr_cyl3","kr_cyl4"] if c in seg.columns]
    if kr_cols:
        kr = seg[kr_cols].to_numpy(dtype=float)
        worst = float(np.nanmin(kr))  # assuming retard is negative
        if abs(worst) >= 4.0:
            sev = "fail"
        elif abs(worst) >= 2.0:
            sev = "warn"
        else:
            sev = "info"
        f.append(Finding(
            code="KNOCK",
            severity=sev,
            title="Knock retard (worst cylinder)",
            detail=f"Worst KR observed {worst:.1f}.",
            evidence={"worst_kr": worst, "kr_columns": kr_cols}
        ))

    # ---- Aircharge cap / load limiting hint ----
    if {"aircharge_pct","aircharge_des_pct"}.issubset(seg.columns):
        ac = _safe_np(seg["aircharge_pct"])
        acd = _safe_np(seg["aircharge_des_pct"])
        diff = acd - ac
        pct = _pct(diff > 5.0)
        if pct > 30:
            f.append(Finding(
                code="AIRCHARGE_MISS",
                severity="warn",
                title="Air Charge not meeting desired",
                detail=f"Air Charge Desired exceeded Actual by >5% for {pct:.1f}% of pull. Can indicate airflow limitation or limiting strategy.",
                evidence={"pct_des_gt_act_5": pct, "diff_p90": float(np.nanpercentile(diff, 90))}
            ))
    return f

def run_cruise_rules(seg: pd.DataFrame) -> list[Finding]:
    f: list[Finding] = []
    if {"stft_pct","ltft_pct"}.issubset(seg.columns):
        st = _safe_np(seg["stft_pct"])
        lt = _safe_np(seg["ltft_pct"])
        st_med = float(np.nanmedian(st))
        lt_med = float(np.nanmedian(lt))
        st_abs_p90 = float(np.nanpercentile(np.abs(st), 90))
        lt_abs_p90 = float(np.nanpercentile(np.abs(lt), 90))
        sev = "info"
        if st_abs_p90 > 10 or lt_abs_p90 > 10:
            sev = "warn"
        if st_abs_p90 > 20 or lt_abs_p90 > 20:
            sev = "fail"
        f.append(Finding(
            code="TRIMS_CRUISE",
            severity=sev,
            title="Fuel trims during steady cruise",
            detail=f"STFT median {st_med:.1f}% (|p90| {st_abs_p90:.1f}%), LTFT median {lt_med:.1f}% (|p90| {lt_abs_p90:.1f}%). High trims suggest intake/evap leaks, MAF scaling, or injector scaling.",
            evidence={"stft_median": st_med, "stft_abs_p90": st_abs_p90, "ltft_median": lt_med, "ltft_abs_p90": lt_abs_p90}
        ))
    if {"afr_actual"}.issubset(seg.columns):
        afr = _safe_np(seg["afr_actual"])
        afr_std = float(np.nanstd(afr))
        if afr_std > 0.6:
            f.append(Finding(
                code="AFR_CRUISE_VAR",
                severity="info",
                title="AFR variability at cruise",
                detail=f"AFR standard deviation {afr_std:.2f}. Large variability can occur with transients; verify the segment is truly steady.",
                evidence={"afr_std": afr_std}
            ))
    return f
