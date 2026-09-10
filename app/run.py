"""
ResQ — one command to bring the whole thing up.

    python app/run.py              build only what is missing, then serve
    python app/run.py --rebuild    force a full rebuild first
    python app/run.py --port 9000  serve on another port

When every artifact already exists this is interactive in well under 10 seconds
(it only starts uvicorn). A full first build is ~10 min (the LSTM + hindcasts).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "static" / "data"
PY = sys.executable


def _load_dotenv():
    """Minimal .env loader (no dependency). Only sets vars not already in the
    environment; ignores comments and blank lines. Never prints values."""
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv()

# (artifact path, builder command, human label)
# Everything here is gitignored and rebuilt on demand from the committed CSVs
# (Raw data/, training_ready_dataset…csv, normal_weather_background.csv) and the
# committed reports/_fragments/*.json.
ARTIFACTS = [
    (ROOT / "data" / "cyclone_clean.parquet",
     [PY, str(ROOT / "scripts" / "build_dataset.py")],
     "cleaned best-track grid (from training_ready_dataset CSV)"),
    (ROOT / "data" / "cyclone_resampled_6h.parquet",
     [PY, "-m", "src.segment_resample"], "6 h resampled grid"),
    (DATA / "engine.pkl",  [PY, str(HERE / "precompute.py")], "forecast engine + replay hindcasts"),
    (DATA / "replay.json", [PY, str(HERE / "precompute.py")], "forecast engine + replay hindcasts"),
    (DATA / "storms.json", [PY, str(ROOT / "scripts" / "build_storms.py")], "2019+ storm tracks"),
    (DATA / "insights.json", [PY, str(ROOT / "scripts" / "build_insights.py")], "metrics + bootstrap CIs"),
    (DATA / "faq.json",    [PY, str(ROOT / "scripts" / "build_faq.py")], "chatbot FAQ"),
    (DATA / "district_history.json",
     [PY, str(ROOT / "scripts" / "build_district_history.py")],
     "district cyclone history (needs GADM download on first run)"),
    (DATA / "ndma_guidance.json",
     [PY, str(ROOT / "scripts" / "build_checklist.py")],
     "NDMA preparedness guidance (Drupal REST harvest; empty state if unreachable)"),
    (DATA / "corpus.json",
     [PY, str(ROOT / "scripts" / "build_corpus.py")], "chatbot retrieval corpus"),
]
# The Official Warnings tab works only by parsing bulletin text the user pastes
# (src/imd_bulletin.py parse_text / explain). There is no download path — IMD has
# no API and the transient operational bulletins are not archived machine-readably.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-browser", action="store_true",
                    help="do not open a web browser on start")
    a = ap.parse_args()

    ran = set()
    for path, cmd, label in ARTIFACTS:
        if not a.rebuild and path.exists():
            continue
        key = tuple(cmd)
        if key in ran:                       # precompute builds two artifacts
            continue
        ran.add(key)
        print(f"[build] {label} …  ({' '.join(cmd[-2:])})")
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            sys.exit(f"build step failed: {' '.join(cmd)}")

    if not ran:
        print("all artifacts present — starting server.")

    import uvicorn
    url = f"http://{a.host}:{a.port}"
    print(f"\n  ResQ dashboard  ->  {url}")
    print("  (open that URL in a browser — do NOT open the .html file directly,\n"
          "   the pages load their data from this server)\n")

    if not a.no_browser:
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run("app.backend.main:app", host=a.host, port=a.port, app_dir=str(ROOT))


if __name__ == "__main__":
    main()
