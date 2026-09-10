# ResQ — phase 10 (close the negation gap in the chat guard)

## Problem

The phase-9 guard caught fabricated **figures, warning colours and future times**
but not **meaning inversion**. An NDMA step like *"DO NOT venture out even when
the winds appear to calm down"* could be paraphrased into its opposite and pass
every check. For safety instructions, paraphrase itself is the risk.

## Fix — verbatim mode for preparedness answers (`app/backend/chat.py`)

**1. Classify.** `_PREP_RE` now also matches "how do I …", "before a cyclone",
"eye of the cyclone", "wind goes quiet/calm/still", "emergency kit", "storm
surge", etc. Any preparedness / precautions / what-should-I-do question routes to
`_prep_verbatim()`.

**2. Assemble, don't generate.**
- `_ndma_rank(question)` keyword-ranks the 51 stored NDMA items (with a small
  cyclone-specific synonym map: quiet → calm/lull/eye, power → mains, …).
- `_groq_select(question, candidates)` sends Groq the id + exact text of each
  candidate and asks for **only** `{"framing": "<one sentence>", "item_ids":
  [...]}`. `temperature 0`, `reasoning_effort: low`. Nothing Groq writes as prose
  reaches the user.
- The reply is built by the program: the chosen framing line, then each item as
  `"<exact stored string>"` on its own line with `— NDMA, <source_url>
  (retrieved <date>)` beneath.

**3. Framing is cosmetic and gated.** `_FRAMING_BANNED` rejects a framing line
that contains any word that could read as an instruction (`go`, `stay`, `avoid`,
`safe`, `out`, `shelter`, `keep`, `ready`, `ensure`, …), a digit, or a colour.
A rejected framing → the fixed default *"Here is what NDMA advises for this
situation:"*; the discard is logged.

**4. Hard assert before sending.** Every `"…"`-quoted line in the assembled reply
is checked against the set of stored NDMA strings **character-for-character**.
Any mismatch → discard, fall back to `_prep_offline()` (same assembly, keyword
ranking instead of Groq). Logged.

**5. Reject unknown ids.** Any `item_id` from Groq that is not in
`ndma_guidance.json` is dropped and logged; if nothing valid remains → offline
assembly.

Modes: `verbatim` (Groq picked the ids), `offline` (keyword ranking picked them,
or Groq failed), `refusal` (no NDMA guidance loaded at all).

## Also — negation check for non-preparedness answers (`_guard`)

`_negation_problem(generation, corpus)`:
- **added negation** — if the generation contains `do not` / `don't` / `never` /
  `avoid` / `must not` and that token does **not** appear in the retrieved
  passages → discard.
- **dropped negation** — for each `do not <phrase>` in the retrieved passages,
  take the salient 1-2 content words; if they appear in the generation without a
  negation within 45 characters before them → discard (`"drops the negation
  before 'venture out' (a cited passage says 'do not venture out even')"`).

## Discard log

Every discard — from either guard — is appended to
`reports/guard_discards.jsonl` (git-ignored, runtime audit trail):

```json
{"ts": "...", "question": "...", "discarded_text": "...", "reason": "..."}
```

## Tests (`tests/test_chat_guard.py`, extended — 14 chat-guard tests, all pass)

- a preparedness answer contains the **exact** NDMA eye-of-the-cyclone string,
  including `DO NOT`
- a synthetic Groq selection whose framing inverts the instruction → the framing
  is dropped, the quoted step is still the exact NDMA string
- a fabricated `item_id` → rejected, logged, never rendered
- every `"…"`-quoted line in a preparedness answer is a stored string
- offline fallback still returns a full, NDMA-sourced answer when Groq selection
  fails
- `_guard` discards an added negation, discards a dropped negation, keeps a
  faithful negation that is in the passage
- `answer()` never raises

`python -m pytest tests/ -v` → **17 passed, 1 skipped** (`real_bulletin.txt`
absent).

## Live run — `POST /api/chat`

**Query:** `what should I do when the wind suddenly goes quiet during a cyclone?`

```json
{
  "answer": "Here is what NDMA advises for this situation:\n\"If the centre of the cyclone is passing directly over your house there will be a lull in the wind and rain lasting for half an hour or so. During this time do not go out; because immediately after that, very strong winds will blow from the opposite direction.\"\n  — NDMA, https://ndma.gov.in/Natural-Hazards/Cyclone/Dos-Donts (retrieved 2026-09-09)\n\"DO NOT venture out even when the winds appear to calm down. The 'eye' of the cyclone might be passing. Winds might intensify and gush again and cause damage. Be safe inside till it is officially announced that the cyclone has passed.\"\n  — NDMA, https://ndma.gov.in/Natural-Hazards/Cyclone/Dos-Donts (retrieved 2026-09-09)",
  "sources": [
    {"source": "NDMA — Cyclone: Do's & Don'ts", "title": "ndma-27", "url": "https://ndma.gov.in/Natural-Hazards/Cyclone/Dos-Donts"},
    {"source": "NDMA — Cyclone: Do's & Don'ts", "title": "ndma-30", "url": "https://ndma.gov.in/Natural-Hazards/Cyclone/Dos-Donts"}
  ],
  "mode": "verbatim",
  "disclaimer": "ResQ model estimate — not an official warning."
}
```

HTTP 200. `mode: verbatim` — gpt-oss-120b narrowed 8 candidates to the two
eye-of-cyclone items (`ndma-27`, `ndma-30`); both quoted lines are the exact
stored NDMA strings, `DO NOT` intact.

## Discards this run

`reports/guard_discards.jsonl` — **one discard**:

```json
{"ts": "2026-09-09T18:10:59Z",
 "question": "what should I do when the wind suddenly goes quiet during a cyclone?",
 "discarded_text": "If the wind suddenly calms, stay inside and wait for official all-clear before venturing out.",
 "reason": "groq framing sentence failed the checks (too long, or contains an instruction / number / colour)"}
```

gpt-oss-120b tried to put a safety instruction ("stay inside and wait … before
venturing out") into the *framing* line — exactly the paraphrase risk. It was
discarded and the fixed framing used; the safety content the user sees is only
the two verbatim NDMA quotes.

## Other live checks

| query | mode | note |
|---|---|---|
| "is it safe to go outside during the eye of the cyclone?" | `verbatim` | returns the exact `DO NOT venture out …` string |
| "does any learned model beat the persistence baseline on track?" | `groq` | guard passed — "no learned track model beats" is in the passages |
| "is a cyclone hitting us today?" | `refusal` | unchanged live-storm refusal |

## Files

**Edited:** `app/backend/chat.py` (verbatim mode, negation guard, discard log,
`GROQ_MODEL` fallbacks `openai/gpt-oss-20b` / `llama-3.3-70b-versatile`),
`tests/test_chat_guard.py` (7 new tests), `.gitignore`
(`reports/guard_discards.jsonl`), the TC-Sentinel HTML (`white-space: pre-wrap`
on `.chat-msg-text` so the assembled multi-line reply renders).
**Created:** `reports/phase10.md`.
