"""
Tests for document upload on the Official Warnings tab:

  src/document_ingest.py   — extraction + magic-byte / size / page guards
  src/imd_bulletin.py      — detect_doc_type() classification
  POST /api/bulletin/upload — the multipart endpoint

Real-document tests (a real single IMD bulletin, a real multi-storm RSMC
report) SKIP cleanly when the fixture is absent — the synthetic fixtures are
NOT substituted for them. The synthetic PDFs (generated with reportlab and
committed) cover the code paths that must always run.

Run:  python -m pytest tests/ -v
"""
from __future__ import annotations

import io
import pathlib

import pytest

from src.document_ingest import DocumentError, extract_document
from src.imd_bulletin import detect_doc_type, find_storm_names

HERE = pathlib.Path(__file__).resolve().parent

REAL_BULLETIN = HERE / "real_bulletin.txt"          # user-supplied, optional
REAL_RSMC = HERE / "real_rsmc_report.pdf"           # user-supplied, optional
NOT_A_BULLETIN_PDF = HERE / "not_a_bulletin.pdf"    # committed synthetic
SYNTH_MULTI_PDF = HERE / "synth_multi_storm.pdf"    # committed synthetic
SYNTH_SINGLE_PDF = HERE / "synth_single_bulletin.pdf"  # committed synthetic
SAMPLE_TXT = HERE / "sample_bulletin.txt"           # committed synthetic


def _client():
    from fastapi.testclient import TestClient
    from app.backend.main import app
    return TestClient(app)


def _upload(client, name, data, storm=None):
    files = {"file": (name, data, "application/octet-stream")}
    form = {"storm": storm} if storm else None
    return client.post("/api/bulletin/upload", files=files, data=form)


# ── extraction guards ─────────────────────────────────────────────────────────
def test_exe_renamed_to_pdf_is_rejected_on_magic_bytes():
    fake = b"MZ\x90\x00\x03" + b"\x00" * 400 + b"This program cannot be run in DOS mode"
    with pytest.raises(DocumentError):
        extract_document("payload.pdf", fake)

    r = _upload(_client(), "payload.pdf", fake)
    assert r.status_code == 400
    assert "MZ" not in r.text  # no echo of file bytes
    body = r.json()
    assert "detail" in body


def test_unsupported_extension_rejected():
    with pytest.raises(DocumentError):
        extract_document("notes.docx", b"PK\x03\x04rest-of-a-zip")


def test_zip_disguised_as_pdf_rejected():
    with pytest.raises(DocumentError):
        extract_document("report.pdf", b"PK\x03\x04" + b"\x00" * 100)


def test_oversize_rejected():
    with pytest.raises(DocumentError):
        extract_document("big.txt", b"a" * (20 * 1024 * 1024 + 1))


def test_empty_upload_is_handled_no_exception_nothing_fabricated():
    with pytest.raises(DocumentError):
        extract_document("empty.txt", b"")

    # endpoint: empty file -> 400, and definitely no parsed bulletin
    r = _upload(_client(), "empty.txt", b"")
    assert r.status_code == 400
    assert "parsed" not in r.json() or r.json().get("parsed") is None


def test_garbage_bytes_named_txt_do_not_crash():
    r = _upload(_client(), "junk.txt", b"\x00\x01\x02\xff\xfe garbage \x00")
    assert r.status_code == 400          # binary content, not plain text
    j = r.json()
    assert j.get("parsed") in (None, {}) if "parsed" in j else True


def test_txt_extraction_shape():
    out = extract_document("b.txt", SAMPLE_TXT.read_bytes())
    assert set(out) == {"text", "page_count", "per_page_text", "chars_extracted"}
    assert out["page_count"] == 1
    assert out["per_page_text"] == [out["text"]]
    assert out["chars_extracted"] > 200


def test_pdf_extraction_retains_pages():
    out = extract_document("m.pdf", SYNTH_MULTI_PDF.read_bytes())
    assert out["page_count"] == 4
    assert len(out["per_page_text"]) == 4
    assert "ALPHA" in out["per_page_text"][1]
    assert "CHARLIE" in out["per_page_text"][3]


# ── classification ───────────────────────────────────────────────────────────
def test_detect_no_text_layer():
    det = detect_doc_type("", page_count=3, chars_extracted=0)
    assert det["doc_type"] == "NO_TEXT_LAYER"
    assert det["storms_found"] == []

    det = detect_doc_type("x. \n  \n .", page_count=5, chars_extracted=4)
    assert det["doc_type"] == "NO_TEXT_LAYER"


def test_detect_not_a_bulletin():
    prose = ("The quarterly review is on Thursday. Lunch will be provided. "
             "Please bring your laptop and the printed agenda. Parking is in "
             "lot C. The facilities team will resurface the car park next month.")
    det = detect_doc_type(prose, page_count=1)
    assert det["doc_type"] == "NOT_A_BULLETIN"
    assert det["storms_found"] == []


def test_detect_not_a_bulletin_pdf_fixture():
    out = extract_document("n.pdf", NOT_A_BULLETIN_PDF.read_bytes())
    det = detect_doc_type(out["text"], out["page_count"],
                          out["chars_extracted"], out["per_page_text"])
    assert det["doc_type"] == "NOT_A_BULLETIN"
    assert det["storms_found"] == []

    r = _upload(_client(), "n.pdf", NOT_A_BULLETIN_PDF.read_bytes())
    assert r.status_code == 200
    j = r.json()
    assert j["doc_type"] == "NOT_A_BULLETIN"
    assert j["parsed"] is None and j["explanation"] is None and j["overlay"] is None
    assert j["storms_found"] == []


def test_detect_multi_storm_from_names():
    text = ("RSMC New Delhi verification report. Cyclonic Storm 'ALPHA' (2020) "
            "formed over the Bay of Bengal. Severe Cyclonic Storm 'BRAVO' (2020) "
            "developed over the Arabian Sea. Very Severe Cyclonic Storm 'CHARLIE' "
            "(2020) made landfall. Maximum sustained wind speed and storm surge "
            "were recorded for each system.")
    det = detect_doc_type(text, page_count=2)
    assert det["doc_type"] == "MULTI_STORM_REPORT"
    assert len(det["storms_found"]) >= 2


def test_detect_multi_storm_by_page_count():
    para = ("Cyclonic Storm 'ONLYONE' formed over the Bay of Bengal and moved "
            "northwestwards. Maximum sustained wind speed of 65 kmph gusting to "
            "75 kmph prevailed near the centre. Storm surge warning was issued "
            "and landfall was expected near the coast. Heavy rainfall occurred. ")
    text = para * 60           # ~15k chars over 40 pages -> real text layer
    det = detect_doc_type(text, page_count=40,
                          chars_extracted=len(text.replace(" ", "")))
    assert det["doc_type"] == "MULTI_STORM_REPORT"


def test_detect_single_bulletin_sample():
    out = extract_document("s.txt", SAMPLE_TXT.read_bytes())
    det = detect_doc_type(out["text"], out["page_count"],
                          out["chars_extracted"], out["per_page_text"])
    assert det["doc_type"] == "SINGLE_BULLETIN"
    assert [s["name"] for s in det["storms_found"]] == ["MANDOUS"]


def test_find_storm_names_page_ranges():
    # page 1 is a contents page (2 names) -> a storm's span is its solo pages
    pages = ["Contents: ALPHA, BRAVO", "Chapter on 'ALPHA' storm details",
             "more 'ALPHA' notes", "Chapter on 'BRAVO' storm"]
    names = find_storm_names("\n".join(pages), pages)
    by = {n["name"]: n for n in names}
    assert by["ALPHA"]["first_page"] == 2 and by["ALPHA"]["last_page"] == 3
    assert by["BRAVO"]["first_page"] == 4 and by["BRAVO"]["last_page"] == 4


# ── endpoint: synthetic single-bulletin PDF ──────────────────────────────────
def test_upload_single_bulletin_pdf_parses():
    r = _upload(_client(), "synth_single_bulletin.pdf", SYNTH_SINGLE_PDF.read_bytes())
    assert r.status_code == 200
    j = r.json()
    assert j["doc_type"] == "SINGLE_BULLETIN"
    assert j["parsed"] is not None
    assert j["parsed"]["system_name"] and "MANDOUS" in j["parsed"]["system_name"]
    assert j["parsed"]["current_position"]["lat"] == pytest.approx(11.5, abs=0.2)
    assert j["explanation"]["lines"]
    assert j["extraction_stats"]["source"] == "uploaded by user"
    # never claims IMD as the source of the file itself
    assert j["extraction_stats"]["doc_type"] == "SINGLE_BULLETIN"


def test_upload_txt_sample_via_endpoint():
    r = _upload(_client(), "sample_bulletin.txt", SAMPLE_TXT.read_bytes())
    assert r.status_code == 200
    j = r.json()
    assert j["doc_type"] == "SINGLE_BULLETIN"
    assert j["parsed"]["district_warnings"], "sample has literal colour+district"


# ── endpoint: synthetic multi-storm PDF, then a section pick ──────────────────
def test_upload_multi_storm_pdf_lists_storms_parses_nothing():
    r = _upload(_client(), "synth_multi_storm.pdf", SYNTH_MULTI_PDF.read_bytes())
    assert r.status_code == 200
    j = r.json()
    assert j["doc_type"] == "MULTI_STORM_REPORT"
    assert len(j["storms_found"]) >= 2
    assert j["parsed"] is None and j["explanation"] is None
    names = {s["name"] for s in j["storms_found"]}
    assert {"ALPHA", "BRAVO", "CHARLIE"} <= names
    for s in j["storms_found"]:
        assert s["first_page"] >= 1 and s["last_page"] >= s["first_page"]


def test_upload_multi_storm_section_pick_parses_only_that_section():
    data = SYNTH_MULTI_PDF.read_bytes()
    r = _upload(_client(), "synth_multi_storm.pdf", data, storm="BRAVO")
    assert r.status_code == 200
    j = r.json()
    assert j["doc_type"] == "MULTI_STORM_REPORT"
    assert j["parsed"] is not None
    # BRAVO's section mentions the Arabian Sea / Gujarat, not CHARLIE
    blob = str(j["parsed"])
    assert "CHARLIE" not in blob

    r2 = _upload(_client(), "synth_multi_storm.pdf", data, storm="NOPE")
    assert r2.status_code == 404


# ── real single IMD bulletin (optional) ──────────────────────────────────────
@pytest.mark.skipif(
    not REAL_BULLETIN.exists(),
    reason="tests/real_bulletin.txt not present — supply a real archived IMD "
           "bulletin. The synthetic fixture is NOT a substitute.",
)
def test_real_single_bulletin_detected_and_core_fields_extracted():
    out = extract_document("real_bulletin.txt", REAL_BULLETIN.read_bytes())
    det = detect_doc_type(out["text"], out["page_count"],
                          out["chars_extracted"], out["per_page_text"])
    assert det["doc_type"] == "SINGLE_BULLETIN", det

    r = _upload(_client(), "real_bulletin.txt", REAL_BULLETIN.read_bytes())
    assert r.status_code == 200
    j = r.json()
    assert j["doc_type"] == "SINGLE_BULLETIN"
    p = j["parsed"]
    assert p and p["system_name"]
    pos = p["current_position"]
    assert pos and 0 < pos["lat"] < 40 and 40 < pos["lon"] < 110
    ci = p["current_intensity"]
    assert ci and (ci.get("category") or ci.get("max_wind_kt") is not None)


# ── real multi-storm RSMC report (optional) ──────────────────────────────────
@pytest.mark.skipif(
    not REAL_RSMC.exists(),
    reason="tests/real_rsmc_report.pdf not present — supply a real RSMC annual "
           "or verification report.",
)
def test_real_multi_storm_report_detected():
    r = _upload(_client(), "real_rsmc_report.pdf", REAL_RSMC.read_bytes())
    assert r.status_code == 200
    j = r.json()
    assert j["doc_type"] == "MULTI_STORM_REPORT"
    assert len(j["storms_found"]) > 1
    assert j["parsed"] is None
