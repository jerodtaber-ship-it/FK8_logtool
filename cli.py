from __future__ import annotations
import argparse
import os

from .schema import load_log, map_columns, basic_cleanup
from .segments import detect_wot_pulls, detect_steady_cruise, slice_segment
from .report import make_reports, print_reports, plot_segment, export_json

def main():
    ap = argparse.ArgumentParser(prog="logtool", description="FK8 Cobb AP log analyzer (starter).")
    ap.add_argument("path", help="Path to Cobb AP log (.csv or .xlsx)")
    ap.add_argument("--plots", action="store_true", help="Write PNG plots for each detected segment.")
    ap.add_argument("--outdir", default="out", help="Output directory for plots.")
    ap.add_argument("--include-cruise", action="store_true", help="Also detect steady-cruise segments for trim analysis.")
    ap.add_argument("--json", action="store_true", help="Also write a machine-readable JSON report to outdir/report.json.")
    args = ap.parse_args()

    raw = load_log(args.path)
    mapped = map_columns(raw)
    df = basic_cleanup(mapped.df)

    if mapped.missing:
        print("NOTE: Missing expected columns (ok if your monitor list differs):")
        for m in mapped.missing:
            print("  -", m)

    segs = []
    segs.extend(detect_wot_pulls(df))
    if args.include_cruise:
        segs.extend(detect_steady_cruise(df))

    if not segs:
        print("No segments detected. Try lowering thresholds in segments.py or verify app_pct/rpm logging.")
        return

    segs = sorted(segs, key=lambda s: s.start_idx)

    reports = make_reports(df, segs)
    print_reports(df, reports)

    if args.plots or args.json:
        os.makedirs(args.outdir, exist_ok=True)

    if args.plots:
        for i, seg in enumerate(segs, start=1):
            seg_df = slice_segment(df, seg)
            outbase = os.path.join(args.outdir, f"segment_{i}_{seg.kind}.png")
            plot_segment(seg_df, outbase)
        print(f"\nPlots written to: {args.outdir}/")

    if args.json:
        outpath = os.path.join(args.outdir, "report.json")
        export_json(reports, outpath)
        print(f"JSON report written to: {outpath}")

if __name__ == "__main__":
    main()
