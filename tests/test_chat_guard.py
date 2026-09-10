"""
Tests for the chatbot's hard guarantees (app/backend/chat.py):
  - live-storm questions are refused
  - the post-generation guard discards an invented warning colour, future time,
    numeric forecast, or meaning-inverting negation
  - PREPAREDNESS runs in verbatim mode: every safety step is an exact NDMA
    string; an inverted step is discarded; an unknown item id is rejected;
    offline fallback still returns a full answer
  - the chatbot never raises — it always returns an answer dict

Run:  python -m pytest tests/ -v
"""
from __future__ import annotations

import pytest

from app.backend import chat

_PASSAGE = [{
    "id": "p1", "source": "reports/limitations.md", "title": "track baseline",
    "text": ("On the 2019+ test the best track baseline is persistence-motion: "
             "33 / 66 / 144 km at 6 / 12 / 24 h. The 80% interval covers 75%."),
}]

# a passage that itself contains a "DO NOT" instruction
_NEG_PASSAGE = [{
    "id": "n1", "source": "NDMA — Cyclone: Do's & Don'ts", "title": "eye",
    "text": ("DO NOT venture out even when the winds appear to calm down. The "
             "'eye' of the cyclone might be passing."),
}]


# ── live-storm ───────────────────────────────────────────────────────────────
def test_live_storm_is_refused():
    r = chat.answer("is there a cyclone forming right now near Odisha?")
    assert r["mode"] == "refusal"
    assert "IMD" in r["answer"]


# ── existing guard checks ────────────────────────────────────────────────────
def test_guard_keeps_a_faithful_generation():
    gen = "The 24-hour track error is 144 km and the 12-hour is 66 km. Sources: 1"
    assert chat._guard(gen, _PASSAGE) is None


def test_guard_discards_invented_warning_colour():
    assert chat._guard("IMD has issued a RED warning for the coast. Sources: 1",
                       _PASSAGE) is not None


def test_guard_discards_invented_future_time():
    assert chat._guard("Landfall is expected around 0230 hrs IST on 12th December.",
                       _PASSAGE) is not None


def test_guard_discards_invented_numeric_forecast():
    assert chat._guard("Peak winds will reach 185 kmph within 24 hours.",
                       _PASSAGE) is not None


# ── NEW: negation / meaning-inversion checks ─────────────────────────────────
def test_guard_discards_added_negation():
    # passage has no negation; generation invents "do not"
    gen = "You should do not use the shelter; stay in your house instead."
    assert chat._guard(gen, _PASSAGE) is not None


def test_guard_discards_dropped_negation():
    # passage says "DO NOT venture out ..."; generation drops the negation
    gen = ("Once the winds appear to calm down you may venture out to check on "
           "neighbours.")
    reason = chat._guard(gen, _NEG_PASSAGE)
    assert reason is not None and "negation" in reason.lower()


def test_guard_keeps_faithful_negation_that_is_in_the_passage():
    gen = "Do not venture out even when the winds appear to calm down."
    assert chat._guard(gen, _NEG_PASSAGE) is None


# ── NEW: verbatim preparedness mode ──────────────────────────────────────────
_EYE_STRING = ("DO NOT venture out even when the winds appear to calm down. The "
               "'eye' of the cyclone might be passing. Winds might intensify and "
               "gush again and cause damage. Be safe inside till it is officially "
               "announced that the cyclone has passed.")


@pytest.mark.skipif(not chat._ndma_items(), reason="NDMA guidance not loaded")
def test_preparedness_answer_contains_exact_eye_of_cyclone_string():
    r = chat.answer("what should I do when the wind suddenly goes quiet during a cyclone?")
    assert r["mode"] in {"verbatim", "offline"}
    assert _EYE_STRING in r["answer"], "the exact NDMA 'DO NOT' string must appear"
    assert "DO NOT" in r["answer"]


@pytest.mark.skipif(not chat._ndma_items(), reason="NDMA guidance not loaded")
def test_preparedness_verbatim_rejects_inverted_step(monkeypatch):
    # Groq 'selects' a real id but also tries to smuggle in an inverted framing
    monkeypatch.setattr(chat, "_groq_select", lambda q, items: {
        "framing": "You can safely go out once the eye arrives:",   # inverts safety
        "item_ids": ["ndma-30"],
    })
    r = chat.answer("what should I do when the wind goes calm mid-cyclone?")
    # the framing is dropped (fails checks); the quoted step is still the exact
    # NDMA string, and it still says DO NOT
    assert _EYE_STRING in r["answer"]
    assert "You can safely go out" not in r["answer"]


@pytest.mark.skipif(not chat._ndma_items(), reason="NDMA guidance not loaded")
def test_preparedness_verbatim_rejects_unknown_item_id(monkeypatch):
    seen = {}
    monkeypatch.setattr(chat, "_groq_select", lambda q, items: {
        "framing": "Here is what NDMA advises:",
        "item_ids": ["ndma-30", "ndma-99999", "totally-made-up"],
    })
    monkeypatch.setattr(chat, "_log_discard",
                        lambda q, t, reason: seen.setdefault("reason", reason))
    r = chat.answer("what should I do when the cyclone eye passes over?")
    assert _EYE_STRING in r["answer"]
    # the fabricated ids never appear
    assert "ndma-99999" not in r["answer"] and "totally-made-up" not in r["answer"]
    assert "reason" in seen and "unknown item id" in seen["reason"]


@pytest.mark.skipif(not chat._ndma_items(), reason="NDMA guidance not loaded")
def test_preparedness_every_quoted_line_is_a_stored_string():
    stored = {it["text"] for it in chat._ndma_items()}
    r = chat.answer("how do I get my house ready before a cyclone?")
    import re
    quoted = re.findall(r'"([^"]+)"', r["answer"])
    assert quoted, "verbatim answer should quote at least one NDMA step"
    for q in quoted:
        assert q in stored, f"quoted line not an exact NDMA string: {q!r}"


@pytest.mark.skipif(not chat._ndma_items(), reason="NDMA guidance not loaded")
def test_offline_fallback_returns_full_answer_when_selection_fails(monkeypatch):
    monkeypatch.setattr(chat, "_groq_select", lambda q, items: None)  # Groq down
    r = chat.answer("what should my family do before a cyclone hits?")
    assert r["mode"] == "offline"
    assert len(r["answer"]) > 120 and r["sources"]
    assert "ndma" in r["sources"][0]["source"].lower()


# ── never-raises ─────────────────────────────────────────────────────────────
def test_answer_never_raises_and_always_has_answer():
    for q in ("", "   ", "what is the capital of France?",
              "tell me about cyclone Amphan", "how do I prepare for a cyclone?",
              "what should I do when the wind suddenly goes quiet?"):
        r = chat.answer(q)
        assert isinstance(r, dict) and isinstance(r.get("answer"), str) and r["answer"]
        assert r.get("mode") in {"offline", "groq", "verbatim", "refusal", "empty"}
