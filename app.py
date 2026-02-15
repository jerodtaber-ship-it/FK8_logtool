from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from logtool.schema import load_log, map_columns, basic_cleanup
from logtool.segments import detect_wot_pulls, detect_steady_cruise, slice_segment
from logtool.report import make_reports, export_json, plot_segment

APP_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(APP_DIR / "templates"))

DATA_ROOT = Path(os.environ.get("LOGTOOL_DATA_DIR", APP_DIR / "data")).resolve()
UPLOAD_DIR = DATA_ROOT / "uploads"
RUNS_DIR = DATA_ROOT / "runs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# If set, restrict upload/analyze UI to people with this key.
# Share links remain accessible without the key.
APP_ACCESS_KEY = os.environ.get("LOGTOOL_ACCESS_KEY", "").strip()

def _clean_old_runs(days: int = 7) -> None:
    """Best-effort cleanup of old run folders."""
    try:
        import time
        cutoff = time.time() - days * 86400
        for d in RUNS_DIR.iterdir():
            if not d.is_dir():
                continue
            try:
                if d.stat().st_mtime < cutoff:
                    # delete files
                    for p in d.rglob("*"):
                        if p.is_file():
                            p.unlink(missing_ok=True)
                    # delete dirs bottom-up
                    for p in sorted([p for p in d.rglob("*") if p.is_dir()], reverse=True):
                        try:
                            p.rmdir()
                        except Exception:
                            pass
                    try:
                        d.rmdir()
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass

app = FastAPI(title="FK8 Cobb Log Tool")

# Serve static PWA assets
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
# Serve run artifacts (plots/json). Directory names include a random share token.
app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    _clean_old_runs()
    return TEMPLATES.TemplateResponse("index.html", {"request": request, "error": None, "needs_key": bool(APP_ACCESS_KEY)})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    include_cruise: Optional[bool] = Form(default=False),
    access_key: Optional[str] = Form(default=None),
):
    _clean_old_runs()

    if APP_ACCESS_KEY:
        if (access_key or "").strip() != APP_ACCESS_KEY:
            return TEMPLATES.TemplateResponse(
                "index.html",
                {"request": request, "error": "Invalid access key.", "needs_key": True},
                status_code=401,
            )

    filename = (file.filename or "upload").strip()
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".csv", ".xlsx", ".xls"]:
        return TEMPLATES.TemplateResponse(
            "index.html",
            {"request": request, "error": "Please upload a .csv or .xlsx log exported from Cobb AP.", "needs_key": bool(APP_ACCESS_KEY)},
            status_code=400,
        )

    run_id = uuid.uuid4().hex  # 32 chars
    share_token = uuid.uuid4().hex  # separate token for sharing
    run_dirname = f"{run_id}_{share_token}"
    run_dir = RUNS_DIR / run_dirname
    run_dir.mkdir(parents=True, exist_ok=True)

    upload_path = UPLOAD_DIR / f"{run_dirname}{ext}"
    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)

    raw = load_log(str(upload_path))
    mapped = map_columns(raw)
    df = basic_cleanup(mapped.df)

    segs = []
    segs.extend(detect_wot_pulls(df))
    if include_cruise:
        segs.extend(detect_steady_cruise(df))
    segs = sorted(segs, key=lambda s: s.start_idx)

    if not segs:
        return TEMPLATES.TemplateResponse(
            "result.html",
            {
                "request": request,
                "run_dirname": run_dirname,
                "filename": filename,
                "missing": mapped.missing,
                "segments": [],
                "plots": [],
                "report_json_url": None,
                "share_url": None,
                "error": "No WOT pulls detected. Try logging a clean 3rd/4th gear pull with pedal to the floor.",
            },
            status_code=200,
        )

    reports = make_reports(df, segs)

    plots = []
    for i, seg in enumerate(segs, start=1):
        seg_df = slice_segment(df, seg)
        outbase = run_dir / f"segment_{i}_{seg.kind}.png"
        plot_segment(seg_df, str(outbase))
        suffixes = ["_boost.png", "_emp_boost.png", "_emp_ratio_vs_rpm.png", "_frp.png", "_throttle.png"]
        for suf in suffixes:
            p = run_dir / (outbase.name.replace(".png", suf))
            if p.exists():
                plots.append({"segment": i, "kind": seg.kind, "name": p.name, "url": f"/runs/{run_dirname}/{p.name}"})

    report_json_path = run_dir / "report.json"
    export_json(reports, str(report_json_path))

    # Save minimal metadata
    meta = {
        "created_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "filename": filename,
        "missing": mapped.missing,
        "include_cruise": bool(include_cruise),
    }
    import json
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    view_segments = []
    for i, rep in enumerate(reports, start=1):
        seg = rep.segment
        seg_df = df.iloc[seg.start_idx:seg.end_idx+1]
        t0, t1 = float(seg_df["time_s"].iloc[0]), float(seg_df["time_s"].iloc[-1])
        r0, r1 = float(seg_df["rpm"].iloc[0]), float(seg_df["rpm"].iloc[-1])
        view_segments.append({
            "idx": i,
            "kind": seg.kind,
            "time_range": f"{t0:.2f}–{t1:.2f}s",
            "rpm_range": f"{r0:.0f}–{r1:.0f}",
            "findings": rep.findings,
            "ranked": rep.ranked_causes or [],
        })

    base_url = str(request.base_url).rstrip("/")
    share_url = f"{base_url}/share/{run_dirname}"

    return TEMPLATES.TemplateResponse(
        "result.html",
        {
            "request": request,
            "run_dirname": run_dirname,
            "filename": filename,
            "missing": mapped.missing,
            "segments": view_segments,
            "plots": plots,
            "report_json_url": f"/runs/{run_dirname}/report.json",
            "share_url": share_url,
            "error": None,
        },
        status_code=200,
    )

@app.get("/share/{run_dirname}", response_class=HTMLResponse)
def share(request: Request, run_dirname: str):
    _clean_old_runs()
    run_dir = (RUNS_DIR / run_dirname).resolve()
    if not run_dir.exists() or not str(run_dir).startswith(str(RUNS_DIR.resolve())):
        return RedirectResponse(url="/", status_code=302)

    report_path = run_dir / "report.json"
    if not report_path.exists():
        return RedirectResponse(url="/", status_code=302)

    import json
    payload = json.loads(report_path.read_text())
    segments = []
    for idx, seg in enumerate(payload, start=1):
        # convert dicts into simple objects for Jinja template attribute access
        findings = [type("F", (), f) for f in seg.get("findings", [])]
        ranked = [type("C", (), c) for c in seg.get("ranked_causes", [])]
        segments.append({
            "idx": idx,
            "kind": seg["segment"]["kind"],
            "findings": findings,
            "ranked": ranked,
        })

    plots = []
    import re
    for p in sorted(run_dir.glob("*.png")):
        m = re.match(r"segment_(\d+)_([A-Z_]+).*\.png$", p.name)
        seg_n = int(m.group(1)) if m else 0
        kind = m.group(2) if m else ""
        plots.append({"segment": seg_n, "kind": kind, "name": p.name, "url": f"/runs/{run_dirname}/{p.name}"})

    # filename from meta if present
    filename = "log"
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            filename = meta.get("filename", filename)
        except Exception:
            pass

    return TEMPLATES.TemplateResponse(
        "share.html",
        {
            "request": request,
            "filename": filename,
            "run_dirname": run_dirname,
            "segments": segments,
            "plots": plots,
            "report_json_url": f"/runs/{run_dirname}/report.json",
        },
        status_code=200,
    )

def main():
    import uvicorn
    uvicorn.run("webapp.app:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
