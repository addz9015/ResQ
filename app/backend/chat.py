"""
Retrieval-only chatbot backend.

Contract (hard):
  * The answer is built ONLY from retrieved passages: the static corpus
    (app/static/data/corpus.json) plus, per request, the bulletin the user has
    pasted. Nothing else.
  * PREPAREDNESS questions run in VERBATIM MODE: the reply is *assembled*, not
    generated. Each safety step is an exact character-for-character copy of a
    stored NDMA item, quoted, with its source URL. Groq may only (a) write one
    short framing sentence and (b) choose which stored item ids to include. It
    cannot rewrite a step, so it cannot invert "DO NOT ...". Before sending,
    every quoted line is re-checked against the stored strings; any mismatch or
    any unknown item id -> discard, fall back to offline retrieval.
  * NON-preparedness answers pass a POST-GENERATION GUARD: discard if the text
    contains a warning colour, a specific future time, a numeric forecast, or a
    NEGATION ("do not", "never", "avoid", ...) that is not in the retrieved
    passages OR drops a negation that a cited passage contains.
  * Live / current-storm questions are refused with a fixed message.
  * Offline keyword retrieval is the DEFAULT. Any Groq failure (missing key,
    timeout, rate-limit, bad response, guard discard) silently falls back to it.
    The chatbot never returns an error.
  * Every discard is appended to reports/guard_discards.jsonl.
  * GROQ_API_KEY and GROQ_MODEL are read from the environment only.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import pathlib
import re

log = logging.getLogger("resq.chat")

HERE = pathlib.Path(__file__).resolve().parent
CORPUS = HERE.parent / "static" / "data" / "corpus.json"
NDMA_GUIDANCE = HERE.parent / "static" / "data" / "ndma_guidance.json"
DISCARD_LOG = HERE.parents[1] / "reports" / "guard_discards.jsonl"

REFUSAL = ("This system works from historical records and published guidance. "
           "For live cyclone warnings, see IMD.")
DISCLAIMER = "ResQ model estimate — not an official warning."
PREP_REFUSAL = (
    "I can only give preparedness steps from a named published source, and none "
    "is loaded in this build. See NDMA's cyclone Do's & Don'ts at "
    "https://ndma.gov.in/Natural-Hazards/Cyclone/Dos-Donts and your State "
    "Disaster Management Authority. (Once NDMA guidance is loaded, this answer "
    "will quote it directly.)")

_LIVE = [
    "live", "current", "currently", "right now", "today", "tonight", "tomorrow",
    "this week", "active now", "real time", "real-time", "realtime", "happening now",
    "latest storm", "ongoing", "forecast for", "will there be", "is there a cyclone",
    "next cyclone", "upcoming",
]
_PREP_RE = re.compile(
    r"\b(prepare|preparation|preparedness|prep\b|safety|safe|evacuat|"
    r"what should i do|what do i do|what to do|how do i|how should i|"
    r"checklist|check list|kit|shelter|precaution|do'?s and don'?ts|"
    r"before (a|an|the) cyclone|during (a|an|the) cyclone|after (a|an|the) cyclone|"
    r"when the cyclone|cyclone (warning|alert)|get ready|ready before|ready for|"
    r"stock up|emergency|eye of the (cyclone|storm)|wind[s]? (go|goes|going) "
    r"(quiet|calm|still|silent)|winds? (suddenly )?(calm|drop|die|quiet)|"
    r"lull in the wind|storm surge|what to pack|what to keep)\b", re.I)

# Guard vocabulary
_COLOUR_RE = re.compile(r"\b(red|orange|yellow|green)\s+"
                        r"(warning|alert|message|colou?r|code|advisory)\b", re.I)
_FUTURE_TIME_RE = re.compile(
    r"\b(\d{3,4}\s*(?:hrs?|hours?)\s*(?:ist|utc)?|"
    r"(?:mid)?night|noon|\d{1,2}\s*(?:am|pm)|"
    r"(?:by|around|on)\s+\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)\b", re.I)

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Currently-available Groq chat model. Override with the GROQ_MODEL env var — no
# code change needed. Secondary models are tried only if the primary errors.
_GROQ_MODEL_DEFAULT = "openai/gpt-oss-120b"
_GROQ_FALLBACK_MODELS = ("openai/gpt-oss-20b", "llama-3.3-70b-versatile")


def _groq_models():
    env = os.environ.get("GROQ_MODEL", "").strip()
    models = [env] if env else [_GROQ_MODEL_DEFAULT]
    for m in _GROQ_FALLBACK_MODELS:
        if m not in models:
            models.append(m)
    return models


_SYS_PROMPT = (
    "You are the ResQ assistant for a historical cyclone research tool.\n"
    "Answer using ONLY the numbered passages below. Follow every rule:\n"
    "1. Use only facts that appear in the passages. Do not add a number, date, "
    "forecast, warning colour, landfall time, or safety instruction that is not "
    "in them.\n"
    "2. If the passages do not contain the answer, say so plainly and stop.\n"
    "3. Never describe a current, forecast or future storm. This tool only "
    "replays past storms and evaluates models.\n"
    "4. Name the source of each claim, e.g. \"(reports/significance.md)\" or "
    "\"(district_history.json)\".\n"
    "5. Plain language, 2-5 sentences.\n"
    "6. End with a line: Sources: <the passage numbers you used>."
)

_vec = None
_mat = None
_passages: list[dict] = []


def _load():
    global _vec, _mat, _passages
    if _vec is not None:
        return
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception:                                # noqa: BLE001
        _passages = []
        return
    if CORPUS.exists():
        _passages = json.loads(CORPUS.read_text(encoding="utf-8")).get("passages", [])
    if not _passages:
        return
    _vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    _mat = _vec.fit_transform([p["title"] + " . " + p["text"] for p in _passages])


_STOP = {"the", "what", "why", "how", "does", "did", "are", "was", "were", "this",
         "that", "with", "for", "and", "you", "your", "his", "her", "our", "about",
         "there", "here", "have", "has", "had", "get", "got", "can", "could",
         "would", "should", "will", "tell", "give", "show", "many", "much", "into",
         "over", "under", "from", "they", "them", "its", "not",
         "who", "whom", "whose", "when", "where", "which", "been", "being",
         "than", "then", "some", "any", "all", "more", "most", "such"}
# light query expansion — link 'how good is it' style words to how the reports
# actually phrase skill / error
_EXPAND = {
    "accurate": ["error", "skill"], "accuracy": ["error", "skill"],
    "error": ["accuracy", "skill"], "good": ["skill", "error"],
    "performance": ["skill", "error"], "reliable": ["calibrat", "coverage"],
    "position": ["track"], "path": ["track"], "strength": ["intensity", "wind"],
    "prepare": ["preparedness"], "prep": ["preparedness"],
}


def _tokens(q: str) -> list[str]:
    # letter-initial terms only — bare numbers / years don't help retrieval and
    # a stray year matching a decade bucket is a classic false positive
    base = [w for w in re.findall(r"[a-z][a-z0-9]{2,}", q.lower()) if w not in _STOP]
    out = list(base)
    for w in base:
        out += _EXPAND.get(w, [])
    return out


def _keyword_scores(question: str, docs: list[dict]) -> dict:
    """IDF-weighted keyword overlap. Rare query words (place names, storm names)
    dominate; common words ('cyclone', 'district') barely count."""
    import math
    words = _tokens(question)
    if not words or not docs:
        return {}
    pats = {w: re.compile(r"\b" + re.escape(w) + r"s?\b") for w in set(words)}
    counts = []
    df = {w: 0 for w in set(words)}
    for p in docs:
        h = (p["title"] + " " + p["title"] + " " + p["text"]).lower()  # title ×2
        c = {w: len(pats[w].findall(h)) for w in set(words)}
        counts.append((p, len(h), c))
        for w, n in c.items():
            if n:
                df[w] += 1
    n = len(docs)
    idf = {w: math.log((n + 1) / (df[w] + 1)) + 1 for w in set(words)}
    out = {}
    for p, ln, c in counts:
        s = sum(idf[w] * c[w] for w in c)
        if s > 0:
            out[p["id"]] = s / (1 + ln / 800)
    return out


def _retrieve(question: str, extra: list[dict], k: int = 5):
    """Returns (passages, confidence). confidence in ~[0, 1]; < ~0.12 means the
    corpus probably has no answer."""
    _load()
    pool = _passages + extra
    if not pool:
        return [], 0.0

    raw_tfidf = {}
    tfidf_top = 0.0
    if _vec is not None and _mat is not None and _passages:
        import numpy as np
        qv = _vec.transform([question])
        sim = (_mat @ qv.T).toarray().ravel()
        tfidf_top = float(sim.max())
        mx = tfidf_top or 1.0
        for i in np.argsort(-sim)[: k * 4]:
            if sim[i] > 0.03:
                raw_tfidf[_passages[i]["id"]] = float(sim[i]) / mx

    raw_kw = _keyword_scores(question, pool)
    kw_top = max(raw_kw.values()) if raw_kw else 0.0
    kw = {i: v / (kw_top or 1.0) for i, v in raw_kw.items()}

    by_id = {p["id"]: p for p in pool}
    ids = set(raw_tfidf) | set(kw)
    merged = {i: 0.6 * raw_tfidf.get(i, 0.0) + 0.4 * kw.get(i, 0.0) for i in ids}

    if re.search(r"\b(accurate|accuracy|error|skill|beat|beats|how good|"
                 r"significan|limitation|underperform|worse|better than)\b",
                 question, re.I):
        for i in list(merged):
            src = by_id[i]["source"]
            if any(s in src for s in ("significance.md", "limitations.md",
                                      "results.md")):
                merged[i] = merged[i] * 1.6 + 0.15

    ranked = sorted(merged.items(), key=lambda kv: -kv[1])
    passages = [by_id[i] for i, _ in ranked[:k]]

    # confidence: raw TF-IDF cosine, and a keyword signal scaled by how MANY of
    # the query's own terms actually land in the top passage (a single rare hit
    # like a stray year is not enough).
    qterms = set(re.findall(r"[a-z][a-z0-9]{2,}", question.lower())) - _STOP
    kw_conf = 0.0
    if ranked and qterms:
        top_txt = (by_id[ranked[0][0]]["title"] + " "
                   + by_id[ranked[0][0]]["text"]).lower()
        n_hit = sum(1 for w in qterms if re.search(r"\b" + re.escape(w) + r"s?\b", top_txt))
        hit_frac = n_hit / len(qterms)
        # need at least two distinct query terms to actually land
        if n_hit >= 2 or (n_hit >= 1 and len(qterms) <= 2):
            kw_conf = min(1.0, (kw_top / (1.0 + kw_top)) * (hit_frac ** 0.5) * 1.7)
    conf = max(tfidf_top * 2.2, kw_conf)
    if any(p["id"] == "bulletin:pasted" for p in passages):
        conf = max(conf, 0.5)
    return passages, round(conf, 3)


def _bulletin_passages(bulletin_text: str | None):
    if not bulletin_text or len(bulletin_text.strip()) < 40:
        return []
    txt = re.sub(r"\s+", " ", bulletin_text).strip()
    return [{
        "id": "bulletin:pasted",
        "source": "bulletin pasted by user",
        "title": "The IMD bulletin you pasted",
        "text": txt[:2500],
    }]


def _offline_answer(passages: list[dict]) -> str:
    if not passages:
        return ("I don't have anything in my records or the loaded guidance that "
                "answers that. I can only answer from the project reports, the "
                "storm archive, district cyclone history and any bulletin you "
                "paste.")
    top = passages[0]
    return top["text"][:700] + ("" if len(top["text"]) <= 700 else " …")


def _try_groq(question: str, passages: list[dict]) -> str | None:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        return None
    try:
        import requests
    except Exception:                                # noqa: BLE001
        return None
    ctx = "\n\n".join(f"[{i + 1}] ({p['source']}) {p['title']}\n{p['text']}"
                      for i, p in enumerate(passages))
    base_body = {
        "messages": [
            {"role": "system", "content": _SYS_PROMPT},
            {"role": "user",
             "content": f"Passages:\n{ctx}\n\nQuestion: {question}"},
        ],
        "temperature": 0.15,
        "max_tokens": 500,
    }
    for model in _groq_models():
        try:
            body = dict(base_body, model=model)
            if any(t in model.lower() for t in ("gpt-oss", "deepseek", "qwen",
                                                "reasoning")):
                # keep chain-of-thought out of `content`
                body["reasoning_format"] = "hidden"
            r = requests.post(_GROQ_URL, json=body, timeout=15,
                              headers={"Authorization": f"Bearer {key}"})
            if r.status_code != 200:
                log.warning("groq %s -> HTTP %s %s", model, r.status_code,
                            r.text[:200])
                continue
            msg = (r.json()["choices"][0]["message"].get("content") or "").strip()
            # strip any stray <think>…</think> if a model ignores reasoning_format
            msg = re.sub(r"(?is)<think>.*?</think>", "", msg).strip()
            if msg:
                return msg
        except Exception as e:                       # noqa: BLE001
            log.warning("groq %s -> %r", model, e)
            continue
    return None


def _log_discard(question: str, text: str, reason: str) -> None:
    """Append every guard discard to reports/guard_discards.jsonl."""
    log.warning("GUARD discarded: %s | Q=%r", reason, question)
    try:
        DISCARD_LOG.parent.mkdir(parents=True, exist_ok=True)
        with DISCARD_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "question": question,
                "discarded_text": text,
                "reason": reason,
            }, ensure_ascii=False) + "\n")
    except Exception as e:                            # noqa: BLE001
        log.warning("could not write discard log: %r", e)


# ── negation handling (meaning-inversion guard) ──────────────────────────────
# the four the spec names, plus unambiguous equivalents
_NEG_TOKENS = ("do not", "don't", "do n't", "never", "avoid",
               "must not", "mustn't", "do not ever")
_NEG_CLAUSE_RE = re.compile(
    r"\b(do not|don't|do n't|never|avoid|must not|mustn't)\s+"
    r"([a-z][a-z'\-]+(?:\s+[a-z][a-z'\-]+){0,3})", re.I)


_FILLER_AFTER_NEG = {"ever", "even", "not", "to", "the", "a", "an", "any", "out",
                     "off", "on", "in", "your", "yourself", "and", "or"}


def _neg_key(phrase: str) -> str:
    """The salient 1-2 content words right after a negation."""
    words = [w for w in phrase.split() if w not in _FILLER_AFTER_NEG or w == "out"]
    return " ".join(words[:2])


def _negation_problem(g: str, corpus: str) -> str | None:
    """g and corpus are lowercased. Returns a reason if g inverts meaning."""
    # (a) g introduces a negation that the passages do not contain
    for neg in _NEG_TOKENS:
        if neg in g and neg not in corpus:
            return f"introduces a negation {neg.strip()!r} absent from the passages"
    # (b) g drops a negation a passage contains: for each "do not <phrase>" in the
    #     corpus, if its salient words appear in g without a negation just before
    for m in _NEG_CLAUSE_RE.finditer(corpus):
        key = _neg_key(m.group(2))
        if len(key) < 4:
            continue
        for gm in re.finditer(re.escape(key), g):
            pre = g[max(0, gm.start() - 45):gm.start()]
            if not any(t in pre for t in _NEG_TOKENS):
                return (f"drops the negation before {key!r} "
                        f"(a cited passage says {m.group(0)!r})")
    return None


def _guard(generation: str, passages: list[dict]) -> str | None:
    """Return a reason string if the generation must be DISCARDED, else None.

    Discard when the text contains a warning colour, a specific future time, a
    numeric forecast, or a meaning-inverting negation that is not supported by
    the retrieved passages."""
    corpus = " ".join(p["text"] for p in passages).lower()
    g = generation.lower()

    if _COLOUR_RE.search(g) and not _COLOUR_RE.search(corpus):
        return "contains a warning colour absent from the passages"

    corpus_ns = corpus.replace(" ", "")
    for m in _FUTURE_TIME_RE.finditer(g):
        phrase = re.sub(r"\s+", " ", m.group(0).lower()).strip()
        if phrase not in corpus and phrase.replace(" ", "") not in corpus_ns:
            return f"contains a specific time {phrase!r} absent from the passages"

    # a measured figure ('N km', 'N kt', 'N%', 'N kmph', 'N hPa') whose numeric
    # value does not appear anywhere in the passages → invented.
    corpus_nums = set(re.findall(r"\d+(?:\.\d+)?", corpus))
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(km|kt|knots|kmph|km/h|hpa|%|per cent)\b",
                         g, re.I):
        if m.group(1) not in corpus_nums:
            return f"contains a figure {m.group(0)!r} absent from the passages"

    neg = _negation_problem(g, corpus)
    if neg:
        return neg
    return None


# ── verbatim mode for preparedness answers ───────────────────────────────────
_NDMA_CACHE: list[dict] | None = None


def _ndma_items() -> list[dict]:
    global _NDMA_CACHE
    if _NDMA_CACHE is not None:
        return _NDMA_CACHE
    _NDMA_CACHE = []
    try:
        d = json.loads(NDMA_GUIDANCE.read_text(encoding="utf-8"))
        if d.get("status") == "ok":
            _NDMA_CACHE = [it for it in d.get("items", [])
                           if isinstance(it.get("text"), str) and it.get("id")]
    except Exception as e:                            # noqa: BLE001
        log.warning("ndma_guidance.json not loadable: %r", e)
    return _NDMA_CACHE


_PREP_EXPAND = {
    "quiet": ["calm", "lull", "eye", "still"], "silent": ["calm", "lull"],
    "still": ["calm", "lull"], "stops": ["calm", "lull"], "stopped": ["calm"],
    "pause": ["lull", "eye"], "dies": ["calm", "lull"], "drops": ["calm"],
    "house": ["home", "building"], "kids": ["children"], "family": ["children"],
    "power": ["electrical", "electricity", "mains"], "food": ["non-perishable"],
    "water": ["drinking"], "leave": ["evacuate"], "documents": ["papers"],
}


def _ndma_rank(question: str, n: int = 8) -> list[dict]:
    """Keyword rank of NDMA items for a preparedness question."""
    items = _ndma_items()
    if not items:
        return []
    base = [w for w in re.findall(r"[a-z][a-z0-9]{2,}", question.lower())
            if w not in _STOP]
    terms = set(base)
    for w in base:
        terms.update(_PREP_EXPAND.get(w, []))
    if not terms:
        return items[:n]
    scored = []
    for it in items:
        t = it["text"].lower()
        hits = sum(1 for w in terms if re.search(r"\b" + re.escape(w) + r"s?\b", t))
        if hits:
            scored.append((hits, it))
    scored.sort(key=lambda x: -x[0])
    return [it for _, it in scored[:n]] or items[:n]


_SELECT_PROMPT = (
    "You are choosing which of the numbered NDMA safety items to show a person, "
    "and writing ONE short framing sentence. You may NOT rewrite, paraphrase, "
    "summarise, shorten or combine any item — the items will be shown to the "
    "user exactly as written, by the program, not by you.\n"
    "Return ONLY compact JSON: {\"framing\": \"<one sentence, <=140 chars, no "
    "safety instructions in it>\", \"item_ids\": [\"<id>\", ...]}\n"
    "Pick the 2-6 items that best fit the question. Use only ids from the list."
)


def _groq_select(question: str, items: list[dict]) -> dict | None:
    """Ask Groq ONLY for {framing, item_ids}. Returns None on any failure."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key or not items:
        return None
    try:
        import requests
    except Exception:                                # noqa: BLE001
        return None
    catalogue = "\n".join(f'{it["id"]}: "{it["text"]}"' for it in items)
    base_body = {
        "messages": [
            {"role": "system", "content": _SELECT_PROMPT},
            {"role": "user",
             "content": f"Items:\n{catalogue}\n\nQuestion: {question}\n\n"
                        "Reply with ONLY the JSON object, nothing before or after."},
        ],
        "temperature": 0.0,
        "max_tokens": 400,
    }
    for model in _groq_models():
        try:
            body = dict(base_body, model=model)
            is_reasoner = any(t in model.lower()
                              for t in ("gpt-oss", "deepseek", "qwen"))
            if is_reasoner:
                body["reasoning_effort"] = "low"
                body["reasoning_format"] = "hidden"
            else:
                body["response_format"] = {"type": "json_object"}
            r = requests.post(_GROQ_URL, json=body, timeout=20,
                              headers={"Authorization": f"Bearer {key}"})
            if r.status_code != 200:
                log.warning("groq select %s -> HTTP %s %s", model,
                            r.status_code, r.text[:200])
                continue
            raw = (r.json()["choices"][0]["message"].get("content") or "").strip()
            raw = re.sub(r"(?is)<think>.*?</think>", "", raw).strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.I).strip()
            m = re.search(r"\{.*\}", raw, re.S)
            if not m:
                log.warning("groq select %s -> no JSON in %r", model, raw[:150])
                continue
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "item_ids" in obj:
                return obj
        except Exception as e:                       # noqa: BLE001
            log.warning("groq select %s -> %r", model, e)
            continue
    return None


_PREP_FRAMING_DEFAULT = "Here is what NDMA advises for this situation:"
# a framing line may not contain anything that could read as an instruction
_FRAMING_BANNED = re.compile(
    r"\b(go|going|goes|gone|stay|stays|leave|leaving|move|moving|do|does|doing|"
    r"don't|avoid|avoids|can|can't|cannot|could|should|shouldn't|must|mustn't|"
    r"may|might|safe|safely|safer|unsafe|out|outside|inside|indoors|outdoors|"
    r"shelter|evacuate|evacuation|open|close|shut|switch|turn|keep|store|stock|"
    r"secure|board|listen|monitor|prepare|pack|carry|bring|use|wait|remain|"
    r"venture|check|drink|boil|charge|tie|fasten|disconnect|unplug|ready|stash|"
    r"ensure|remember|make sure|need to|have to)\b", re.I)


def _prep_verbatim(question: str) -> dict:
    """Assemble a preparedness answer from EXACT NDMA strings. Groq may only
    pick ids + a framing line; every quoted line is re-verified char-for-char."""
    items = _ndma_items()
    if not items:
        return {"answer": PREP_REFUSAL, "mode": "refusal",
                "sources": [{"source": "NDMA",
                             "title": "NDMA cyclone Do's & Don'ts — not loaded"}]}
    by_id = {it["id"]: it for it in items}
    stored = {it["text"] for it in items}

    candidates = _ndma_rank(question, n=8)

    sel = _groq_select(question, candidates)
    if sel is None:
        # Groq unavailable / failed -> offline retrieval assembly
        return _prep_offline(question, candidates)

    want = sel.get("item_ids") or []
    unknown = [i for i in want if i not in by_id]
    if unknown:
        _log_discard(question, json.dumps(sel, ensure_ascii=False),
                     f"groq selection returned unknown item id(s) {unknown}")
    picked = [i for i in want if i in by_id]
    if not picked:
        return _prep_offline(question, candidates)

    fr = str(sel.get("framing", "")).strip().strip('"')
    fr_bad = (len(fr) > 140 or not fr
              or _FRAMING_BANNED.search(fr)
              or re.search(r"\d", fr)
              or _COLOUR_RE.search(fr))
    if fr_bad and fr:
        _log_discard(question, fr, "groq framing sentence failed the checks "
                     "(too long, or contains an instruction / number / colour)")
    framing = fr if not fr_bad else _PREP_FRAMING_DEFAULT
    chosen_ids = picked

    lines = [framing]
    used = []
    for i in chosen_ids:
        it = by_id.get(i)
        if not it:
            continue
        lines.append(f'"{it["text"]}"')
        lines.append(f'  — NDMA, {it.get("source_url", "https://ndma.gov.in")}'
                     f' (retrieved {it.get("retrieved_on", "")})')
        used.append(it)
    response = "\n".join(lines)

    # HARD ASSERT: every quoted line is an exact stored string
    for quoted in re.findall(r'"([^"]+)"', response):
        if quoted not in stored:
            _log_discard(question, response,
                         "assembled quote did not match a stored NDMA string "
                         "character-for-character")
            return _prep_offline(question, candidates)

    if not used:
        return _prep_offline(question, candidates)

    return {
        "answer": response,
        "sources": [{"source": it.get("source_name", "NDMA"),
                     "title": it["id"], "url": it.get("source_url")}
                    for it in used],
        "mode": "verbatim",
        "disclaimer": DISCLAIMER,
    }


def _prep_offline(question: str, candidates: list[dict] | None = None) -> dict:
    items = candidates if candidates is not None else _ndma_rank(question, n=6)
    items = items[:6]
    if not items:
        return {"answer": PREP_REFUSAL, "mode": "refusal",
                "sources": [{"source": "NDMA",
                             "title": "NDMA cyclone Do's & Don'ts — not loaded"}]}
    lines = [_PREP_FRAMING_DEFAULT]
    for it in items:
        lines.append(f'"{it["text"]}"')
        lines.append(f'  — NDMA, {it.get("source_url", "https://ndma.gov.in")}')
    return {
        "answer": "\n".join(lines),
        "sources": [{"source": it.get("source_name", "NDMA"), "title": it["id"],
                     "url": it.get("source_url")} for it in items],
        "mode": "offline",
        "disclaimer": DISCLAIMER,
    }


def answer(question: str, bulletin_text: str | None = None) -> dict:
    q = (question or "").strip()
    if not q:
        return {"answer": "Ask me something about how the models work, a past "
                          "storm, your district's cyclone history, or a bulletin "
                          "you've pasted.", "sources": [], "mode": "empty"}

    ql = q.lower()
    if any(w in ql for w in _LIVE):
        return {"answer": REFUSAL, "sources": [], "mode": "refusal"}

    extra = _bulletin_passages(bulletin_text)
    passages, conf = _retrieve(q, extra)
    for e in extra:
        if e["id"] not in {p["id"] for p in passages}:
            passages = [e] + passages[:4]

    # Preparedness questions -> VERBATIM MODE. The reply is assembled from exact
    # NDMA strings; Groq may only pick item ids + a framing line. It cannot
    # rewrite a step, so it cannot invert "DO NOT ...".
    if _PREP_RE.search(ql):
        return _prep_verbatim(q)

    # nothing in the corpus is a plausible match -> say so, don't force a passage
    if conf < 0.15:
        return {"answer": (
            "I don't have anything in the project reports, the storm archive, "
            "district cyclone history or the loaded NDMA guidance that answers "
            "that. I can only answer from those sources and any bulletin you "
            "paste."), "sources": [], "mode": "offline"}

    sources = [{"source": p["source"], "title": p["title"]} for p in passages]

    groq = _try_groq(q, passages)
    if groq:
        reason = _guard(groq, passages)
        if reason:
            _log_discard(q, groq, reason)
        else:
            return {"answer": groq, "sources": sources, "mode": "groq",
                    "disclaimer": DISCLAIMER}
    return {"answer": _offline_answer(passages), "sources": sources,
            "mode": "offline", "disclaimer": DISCLAIMER}


def _is_ndma(p: dict) -> bool:
    s = (p.get("source") or "").lower()
    return "ndma" in s or s.startswith("ndma")


def status() -> dict:
    _load()
    return {
        "corpus_passages": len(_passages),
        "ndma_items": len(_ndma_items()),
        "groq_key_present": bool(os.environ.get("GROQ_API_KEY", "").strip()),
        "groq_model": _groq_models()[0],
        "default_mode": "offline keyword retrieval",
        "preparedness_mode": "verbatim (assembled from exact NDMA strings)",
        "discard_log": str(DISCARD_LOG),
    }
