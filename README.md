# FK8 Cobb Log Tool (Starter)

This is a minimal, working starting point for:
- Loading Cobb Accessport logs (CSV or XLSX)
- Normalizing column names to a canonical schema
- Auto-detecting WOT pulls
- Running a few high-signal diagnostics (boost tracking, throttle closure, FRP tracking, AFR tracking, KR)
- Generating a simple text report + optional plots

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
logtool examples/datalog102.xlsx --plots
```

## What to edit first
- `logtool/schema.py`: Cobb FK8 column mapping + unit normalization
- `logtool/segments.py`: pull detection logic (WOT / steady-state)
- `logtool/rules.py`: diagnostic rules and thresholds
- `logtool/report.py`: how results are presented

## Notes
- Cobb monitor names vary. The mapper supports exact matches plus a small fuzzy fallback.
- Wastegate position polarity can differ. A stub detector is included; improve it with more labeled data.

## New in v0.3
- EMP:Boost ratio plotted vs RPM + binned-by-RPM finding
- Ranked likely causes + concrete next steps for each WOT pull
- Optional JSON output for UI integration

Run:
```bash
logtool examples/datalog102.xlsx --plots --json
```

## Web app (FastAPI)

Run locally:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

# Start the web app (auto-reload)
logtool-web
```

Then open:
- http://127.0.0.1:8000

Uploads and run artifacts (plots/report.json) are stored under `webapp/data/`.
You can change the data directory by setting:
- `LOGTOOL_DATA_DIR=/path/to/data`

## PWA (Install on iPhone)

This web app includes a PWA manifest + service worker.

### Install (iPhone)
1. Start the server (see Web app section) and access it in **Safari** over **HTTPS**.
   - iOS requires HTTPS for service workers and “Add to Home Screen” to behave like an app.
2. In Safari, tap **Share** → **Add to Home Screen**.
3. Launch “FK8 LogTool” from your Home Screen (standalone mode).

### HTTPS tip (local dev)
- If you want to test on your phone from your home Wi‑Fi, use an HTTPS tunnel (e.g., a tunnel tool) that gives you an https:// URL pointing at your laptop's http://127.0.0.1:8000.
- Or run uvicorn with a local certificate (mkcert) and access `https://<your-laptop-ip>:8000`.

## Share with tuner (v0.6)

After analysis, the result page shows a **Share link** like:
`https://your-app.com/share/<run_id>_<share_token>`

That link is public and unguessable; it shows the report + plots without requiring the access key.

### Protect uploads with an access key
Set:
- `LOGTOOL_ACCESS_KEY=your-secret`

Then upload/analyze requires the key, while share links still work.

### Retention
Best-effort cleanup removes run folders older than ~7 days.
