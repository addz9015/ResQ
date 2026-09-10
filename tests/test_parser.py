"""
Parser tests for src/imd_bulletin.py — parse_text() and explain().

The headline test runs against a REAL archived IMD bulletin at
tests/real_bulletin.txt (supplied separately). If that file is absent the test
SKIPS with a clear message — the synthetic tests/sample_bulletin.txt is NOT
substituted, because "the synthetic fixture parses" is not evidence that the
parser handles a real bulletin.

Run:  python -m pytest tests/ -v
"""
from __future__ import annotations

import pathlib

import pytest

from src.imd_bulletin import explain, parse_text

HERE = pathlib.Path(__file__).resolve().parent
REAL = HERE / "real_bulletin.txt"
SAMPLE = HERE / "sample_bulletin.txt"

_NONSENSE = (
    "Meeting notes: the quarterly review is on Thursday. Lunch will be provided. "
    "Please bring your laptops and the printed agenda. Parking is in lot C."
)


# ── the real-bulletin test ────────────────────────────────────────────────────
@pytest.mark.skipif(
    not REAL.exists(),
    reason="tests/real_bulletin.txt not present — supply a real archived IMD "
           "bulletin to run this test. The synthetic fixture is NOT a substitute.",
)
def test_parse_real_bulletin_extracts_core_fields():
    raw = REAL.read_text(encoding="utf-8")
    p = parse_text(raw)

    assert p["system_name"], "no system name extracted from the real bulletin"

    pos = p["current_position"]
    assert pos and isinstance(pos.get("lat"), (int, float)) \
        and isinstance(pos.get("lon"), (int, float)), \
        f"no usable position extracted: {pos!r}"
    assert 0 < pos["lat"] < 40 and 40 < pos["lon"] < 110, \
        f"position out of the North Indian Ocean box: {pos!r}"

    ci = p["current_intensity"]
    assert ci and (ci.get("category") or ci.get("max_wind_kt") is not None), \
        f"no intensity extracted: {ci!r}"

    dw = p["district_warnings"]
    assert isinstance(dw, list) and len(dw) >= 1, \
        "expected at least one district warning from the real bulletin"
    w = dw[0]
    assert w["district"] and w["colour"] in {"red", "orange", "yellow", "green"}, \
        f"district warning missing district or colour: {w!r}"

    # explain() must not invent — every line is a string, nulls become the marker
    lines = explain(p)["lines"]
    assert all(isinstance(v, str) and v for v in lines.values())


# ── the never-fabricate test (always runs) ────────────────────────────────────
def test_nonsense_yields_all_null_and_fabricates_nothing():
    p = parse_text(_NONSENSE)

    assert p["issued_at"] is None
    assert p["system_name"] is None
    assert p["current_position"] is None
    assert p["current_intensity"] is None
    assert p["movement"] is None
    assert p["forecast_track"] is None
    assert p["district_warnings"] == []          # never a fabricated warning
    assert p["landfall_estimate"] is None

    lines = explain(p)["lines"]
    marker = "not stated in this bulletin"
    for field in ("issued_at", "system_name", "current_position",
                  "current_intensity", "movement", "forecast_track",
                  "landfall_estimate"):
        assert marker in lines[field], f"{field}: expected the null marker"


def test_empty_input_is_safe():
    for bad in ("", "   ", None):
        p = parse_text(bad)  # type: ignore[arg-type]
        assert p["system_name"] is None
        assert p["district_warnings"] == []


# ── the synthetic fixture is exercised, but labelled as synthetic ─────────────
@pytest.mark.skipif(not SAMPLE.exists(), reason="sample fixture missing")
def test_synthetic_fixture_smoke_only():
    """Synthetic fixture — proves the code path runs, NOT that real bulletins
    parse. `test_parse_real_bulletin_extracts_core_fields` is the real check."""
    p = parse_text(SAMPLE.read_text(encoding="utf-8"))
    assert p["system_name"] and p["current_position"]
    # nothing invented: raw phrases are echoed for every district warning
    for w in p["district_warnings"]:
        assert w.get("raw_phrase")
