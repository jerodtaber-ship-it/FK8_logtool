from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class RankedCause:
    cause: str
    score: float
    rationale: str
    next_steps: list[str]

def _has(code: str, findings) -> bool:
    return any(f.code == code for f in findings)

def _get(code: str, findings):
    for f in findings:
        if f.code == code:
            return f
    return None

def rank_causes_wot(seg: pd.DataFrame, findings) -> list[RankedCause]:
    """
    Lightweight, explainable scorer that converts findings + a few signals into:
      - ranked likely causes
      - concrete next steps

    This is intentionally heuristic (v1). As you label cases, you can replace this with ML.
    """
    scores = {
        "torque_or_traction_limiting": 0.0,
        "turbo_component_protection_limiting": 0.0,
        "turbine_choking_or_exhaust_restriction": 0.0,
        "boost_leak_post_turbo": 0.0,
        "pre_turbine_exhaust_leak": 0.0,
        "wastegate_mechanical_or_control_issue": 0.0,
        "fueling_system_limit": 0.0,
    }

    # Base: underboost present?
    under = _get("UNDERBOOST", findings)
    if under:
        for k in scores:
            scores[k] += 0.5

    # Throttle closure points strongly to limiters
    if _has("THROTTLE_CLOSURE", findings):
        scores["torque_or_traction_limiting"] += 4.0
    if _has("THROTTLE_OK", findings) and under:
        scores["torque_or_traction_limiting"] -= 1.0

    # Component protection
    if _has("COMP_PROTECT", findings):
        scores["turbo_component_protection_limiting"] += 4.0

    # Fueling
    frp = _get("FRP_DROP", findings)
    if frp:
        scores["fueling_system_limit"] += 3.0
    if _has("HPFP_AT_LIMIT", findings):
        scores["fueling_system_limit"] += 2.0
    if _has("AFR_MISS", findings):
        scores["fueling_system_limit"] += 1.5
    if _has("FRP_OK", findings) and _has("AFR_OK", findings):
        scores["fueling_system_limit"] -= 1.0

    # EMP ratio — high implies turbine choking / restriction / pre-turb issues
    emp = _get("EMP_RATIO", findings)
    if emp:
        ratio = emp.evidence.get("emp_boost_gauge_ratio_median")
        if ratio is not None:
            if ratio > 2.5:
                scores["turbine_choking_or_exhaust_restriction"] += 4.0
                scores["pre_turbine_exhaust_leak"] += 1.0
            elif ratio > 2.2:
                scores["turbine_choking_or_exhaust_restriction"] += 3.0
                scores["pre_turbine_exhaust_leak"] += 0.7
            elif ratio < 1.8:
                # lower EMP ratio makes restriction less likely; consider boost leak more
                scores["turbine_choking_or_exhaust_restriction"] -= 0.5
                scores["boost_leak_post_turbo"] += 1.0

    # Wastegate saturation hints
    if _has("WG_SAT_CLOSED", findings) and under:
        # Gate "closed" yet still underboosting: flow ceiling/leak
        scores["turbine_choking_or_exhaust_restriction"] += 1.2
        scores["boost_leak_post_turbo"] += 1.2
        scores["pre_turbine_exhaust_leak"] += 1.0
        scores["wastegate_mechanical_or_control_issue"] -= 0.5

    # WG polarity uncertain / control issue bucket
    if _has("WG_POLARITY_UNCERTAIN", findings):
        scores["wastegate_mechanical_or_control_issue"] += 2.5

    # Use a small additional signal: pre-throttle vs MAP (helps separate throttle/restriction/leak a bit)
    # If throttle is open and pre-throttle pressure is high relative to MAP, can imply restriction between sensors
    if {"pre_throttle_p_act_psi","map_psi"}.issubset(seg.columns):
        p_pre = pd.to_numeric(seg["pre_throttle_p_act_psi"], errors="coerce").to_numpy(dtype=float)
        p_map = pd.to_numeric(seg["map_psi"], errors="coerce").to_numpy(dtype=float)
        d = np.nanmedian(p_pre - p_map)
        if np.isfinite(d):
            if d > 2.0:
                scores["boost_leak_post_turbo"] += 1.0
                scores["turbine_choking_or_exhaust_restriction"] += 0.5

    # Clamp and sort
    out = []
    for k, v in scores.items():
        out.append((k, float(max(0.0, v))))

    out.sort(key=lambda x: x[1], reverse=True)

    def steps_for(cause: str) -> list[str]:
        if cause == "torque_or_traction_limiting":
            return [
                "Log again including any available torque/traction/limit flags (if accessible) and verify pedal vs throttle closure alignment.",
                "Repeat the same pull with TC fully disabled (where safe/legal) to see if throttle closure disappears.",
                "Check IAT/MAT/ECT during the event; thermal strategies can trigger closure."
            ]
        if cause == "turbo_component_protection_limiting":
            return [
                "Compare 'Turbo Maximum Boost - Component Protection' vs boost target across RPM; if it sits below target, reduce target/load in that region.",
                "Inspect IAT/MAT and coolant temps; component protection often correlates with heat.",
                "Review intake/exhaust/turbo sizing vs requested pressure ratio."
            ]
        if cause == "turbine_choking_or_exhaust_restriction":
            return [
                "Compute EMP:Boost by RPM; if it climbs sharply up top, reduce high-RPM boost target or improve exhaust flow/turbine capacity.",
                "Check for exhaust restrictions (collapsed flex, clogged cat) and verify downpipe/cat health.",
                "Verify turbo/wastegate hardware and manifold to turbo sealing; high EMP can also occur with pre-turb leaks."
            ]
        if cause == "boost_leak_post_turbo":
            return [
                "Perform a charge-pipe/IC pressure test to your typical peak boost (e.g., 20–25 psi) and listen/soap-test for leaks.",
                "Review clamps/couplers, BPV connections, and intercooler end-tanks; re-log after fixing any leak.",
                "At steady cruise, look for elevated positive trims (STFT/LTFT) that support unmetered air or leaks."
            ]
        if cause == "pre_turbine_exhaust_leak":
            return [
                "Inspect manifold-to-turbo gasket, turbo inlet flange, and any welds for soot trails or ticking noise on cold start.",
                "Smoke test the exhaust pre-turb if possible; re-torque hardware after heat-cycling.",
                "Compare EMP behavior before/after repair; leaks can alter boost response and EMP."
            ]
        if cause == "wastegate_mechanical_or_control_issue":
            return [
                "Verify wastegate actuator preload and that the arm moves freely through full travel.",
                "Confirm the logged WG position meaning (is higher more open or more closed?) and check for sensor scaling issues.",
                "If possible, run a controlled test at lower boost target to see if WG position responds predictably."
            ]
        if cause == "fueling_system_limit":
            return [
                "Check FRP Actual vs Desired at peak load; if shortfall grows with RPM, reduce load/boost or revise HPFP targets.",
                "Inspect LPFP supply (if you can log it) and fuel quality; verify injector characterization if AFR misses commanded.",
                "Re-log with consistent fuel and temps to confirm repeatability."
            ]
        return ["Re-log with consistent conditions and add any missing monitors relevant to this hypothesis."]

    # Build rationales from the findings
    for cause, score in out[:3]:
        rationale_bits = []
        if under:
            rationale_bits.append("Underboost is present.")
        if cause == "torque_or_traction_limiting" and _has("THROTTLE_CLOSURE", findings):
            rationale_bits.append("Throttle closes during WOT (classic limiter/traction pattern).")
        if cause == "turbo_component_protection_limiting" and _has("COMP_PROTECT", findings):
            rationale_bits.append("Component protection ceiling sits below target for part of the pull.")
        if cause == "turbine_choking_or_exhaust_restriction" and emp:
            rationale_bits.append("EMP:Boost ratio is elevated.")
        if cause in ("boost_leak_post_turbo","pre_turbine_exhaust_leak") and _has("WG_SAT_CLOSED", findings):
            rationale_bits.append("WG appears near 'closed' extreme yet boost target is missed (points to leaks/flow ceiling).")
        if cause == "fueling_system_limit" and (frp or _has("AFR_MISS", findings)):
            rationale_bits.append("Fuel pressure/AFR tracking indicates potential fueling headroom issues.")
        if cause == "wastegate_mechanical_or_control_issue" and _has("WG_POLARITY_UNCERTAIN", findings):
            rationale_bits.append("WG position semantics look inconsistent; control/interpretation needs verification.")
        rationale = " ".join(rationale_bits) if rationale_bits else "Scored by heuristic signals from this pull."

        out_obj = RankedCause(
            cause=cause,
            score=score,
            rationale=rationale,
            next_steps=steps_for(cause)
        )
        # Only keep meaningful entries
        if score >= 1.0:
            return_list = []
    # Return all causes with score >=1, capped at 5
    ranked = []
    for cause, score in out:
        if score >= 1.0:
            ranked.append(RankedCause(cause=cause, score=score, rationale="Scored by heuristic signals.", next_steps=steps_for(cause)))
        if len(ranked) >= 5:
            break

    # Improve rationale for top 3
    for i in range(min(3, len(ranked))):
        cause = ranked[i].cause
        # reuse detailed rationale
        ranked[i] = RankedCause(cause=cause, score=ranked[i].score, rationale=rank_causes_wot(seg, findings=[])[0].rationale if False else ranked[i].rationale, next_steps=ranked[i].next_steps)
    # overwrite top-3 with detailed ones computed above
    detailed_top3 = []
    for cause, score in out[:3]:
        if score < 1.0:
            continue
        rationale_bits = []
        if under:
            rationale_bits.append("Underboost is present.")
        if _has("THROTTLE_CLOSURE", findings):
            rationale_bits.append("Throttle closure detected.")
        if _has("COMP_PROTECT", findings):
            rationale_bits.append("Component protection indicated.")
        empf = emp
        if empf:
            r = empf.evidence.get("emp_boost_gauge_ratio_median")
            if r is not None:
                rationale_bits.append(f"EMP:Boost≈{r:.2f}.")
        if _has("WG_SAT_CLOSED", findings):
            rationale_bits.append("WG near closed extreme during underboost.")
        if frp:
            rationale_bits.append("FRP shortfall present.")
        rationale = " ".join(rationale_bits) if rationale_bits else "Scored by heuristic signals from this pull."
        detailed_top3.append(RankedCause(cause=cause, score=score, rationale=rationale, next_steps=steps_for(cause)))
    # Replace beginning of ranked list with detailed
    if detailed_top3:
        # keep unique order
        seen=set()
        merged=[]
        for rc in detailed_top3 + ranked:
            if rc.cause in seen: 
                continue
            seen.add(rc.cause)
            merged.append(rc)
        ranked = merged[:5]

    return ranked
