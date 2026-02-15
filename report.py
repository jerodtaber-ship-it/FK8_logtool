from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .segments import Segment, slice_segment
from .rules import run_wot_rules, run_cruise_rules
from .diagnosis import rank_causes_wot

@dataclass
class SegmentReport:
    segment: Segment
    findings: list
    ranked_causes: list | None = None

def make_reports(df: pd.DataFrame, segments: list[Segment]) -> list[SegmentReport]:
    reps = []
    for seg in segments:
        seg_df = slice_segment(df, seg)
        if seg.kind == "WOT_PULL":
            findings = run_wot_rules(seg_df)
            ranked = rank_causes_wot(seg_df, findings)
        elif seg.kind == "CRUISE":
            findings = run_cruise_rules(seg_df)
            ranked = None
        else:
            findings = []
            ranked = None
        reps.append(SegmentReport(segment=seg, findings=findings, ranked_causes=ranked))
    return reps

def print_reports(df: pd.DataFrame, reports: list[SegmentReport]) -> None:
    for i, rep in enumerate(reports, start=1):
        seg_df = df.iloc[rep.segment.start_idx:rep.segment.end_idx+1]
        t0, t1 = float(seg_df["time_s"].iloc[0]), float(seg_df["time_s"].iloc[-1])
        r0, r1 = float(seg_df["rpm"].iloc[0]), float(seg_df["rpm"].iloc[-1])
        print(f"\n=== Segment {i}: {rep.segment.kind}  idx[{rep.segment.start_idx}:{rep.segment.end_idx}]  time {t0:.2f}-{t1:.2f}s  rpm {r0:.0f}-{r1:.0f} ===")
        for f in rep.findings:
            print(f"- [{f.severity.upper()}] {f.title}: {f.detail}")

        if rep.segment.kind == "WOT_PULL" and rep.ranked_causes:
            print("\n  Likely causes (ranked) + next steps:")
            for j, c in enumerate(rep.ranked_causes[:3], start=1):
                print(f"  {j}) {c.cause} (score {c.score:.1f}) — {c.rationale}")
                for step in c.next_steps[:3]:
                    print(f"     - {step}")

def export_json(reports: list[SegmentReport], outpath: str) -> None:
    def ser(obj):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)
    payload = []
    for r in reports:
        payload.append({
            "segment": {"kind": r.segment.kind, "start_idx": r.segment.start_idx, "end_idx": r.segment.end_idx},
            "findings": [f.__dict__ for f in r.findings],
            "ranked_causes": [c.__dict__ for c in (r.ranked_causes or [])],
        })
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2, default=ser)

def plot_segment(seg_df: pd.DataFrame, outbase: str) -> None:
    # Boost overlay
    if {"time_s","boost_actual_psi","boost_target_psi"}.issubset(seg_df.columns):
        plt.figure()
        plt.plot(seg_df["time_s"], seg_df["boost_target_psi"], label="Boost Target (psi)")
        plt.plot(seg_df["time_s"], seg_df["boost_actual_psi"], label="Boost Actual (psi)")
        plt.xlabel("Time (s)")
        plt.ylabel("Boost (psi)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outbase.replace(".png", "_boost.png"), dpi=160)
        plt.close()

    # EMP overlay
    if {"time_s","emp_psi","boost_actual_psi"}.issubset(seg_df.columns):
        plt.figure()
        plt.plot(seg_df["time_s"], seg_df["emp_psi"], label="Exhaust Manifold Pressure (psi)")
        plt.plot(seg_df["time_s"], seg_df["boost_actual_psi"], label="Boost Actual (psi)")
        plt.xlabel("Time (s)")
        plt.ylabel("Pressure (psi)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outbase.replace(".png", "_emp_boost.png"), dpi=160)
        plt.close()

    # EMP:Boost ratio by RPM (if present)
    if {"rpm","emp_psi","boost_actual_psi"}.issubset(seg_df.columns):
        rpm = pd.to_numeric(seg_df["rpm"], errors="coerce").to_numpy(dtype=float)
        emp = pd.to_numeric(seg_df["emp_psi"], errors="coerce").to_numpy(dtype=float)
        bst = pd.to_numeric(seg_df["boost_actual_psi"], errors="coerce").to_numpy(dtype=float)
        m = (bst > 2.0) & np.isfinite(rpm) & np.isfinite(emp) & np.isfinite(bst)
        if np.sum(m) > 20:
            ratio = emp[m] / bst[m]
            plt.figure()
            plt.plot(rpm[m], ratio, label="EMP/Boost (gauge)")
            plt.xlabel("RPM")
            plt.ylabel("Ratio")
            plt.legend()
            plt.tight_layout()
            plt.savefig(outbase.replace(".png", "_emp_ratio_vs_rpm.png"), dpi=160)
            plt.close()

    # FRP overlay
    if {"time_s","frp_actual_psi","frp_target_psi"}.issubset(seg_df.columns):
        plt.figure()
        plt.plot(seg_df["time_s"], seg_df["frp_target_psi"], label="FRP Target (psi)")
        plt.plot(seg_df["time_s"], seg_df["frp_actual_psi"], label="FRP Actual (psi)")
        plt.xlabel("Time (s)")
        plt.ylabel("Fuel Rail Pressure (psi)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outbase.replace(".png", "_frp.png"), dpi=160)
        plt.close()

    # Throttle & pedal
    if {"time_s","app_pct","throttle_pct"}.issubset(seg_df.columns):
        plt.figure()
        plt.plot(seg_df["time_s"], seg_df["app_pct"], label="Pedal (%)")
        plt.plot(seg_df["time_s"], seg_df["throttle_pct"], label="Throttle (%)")
        plt.xlabel("Time (s)")
        plt.ylabel("Percent")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outbase.replace(".png", "_throttle.png"), dpi=160)
        plt.close()
