# Hand-filling `ndma_guidance.json`

If `scripts/build_checklist.py` cannot retrieve NDMA guidance, a human must transcribe it:

1. Open NDMA's *Cyclone: Do's & Don'ts* (https://ndma.gov.in/Natural-Hazards/Cyclone/Dos-Donts) and *Cyclone* (https://ndma.gov.in/Natural-Hazards/Cyclone).
2. Copy `TEMPLATE.json` to `../ndma_guidance.json`.
3. For **each** published bullet, add an item. `text` must be the exact sentence — no paraphrase, nothing from memory, nothing from an LLM. Fill `source_name`, `source_url`, `retrieved_on` (today's date), and `time_band` from the section it appears under:
   - *Before the cyclone season* / *When the cyclone starts* / *Emergency Kit* -> `now`
   - *When your area is under cyclone warning* / *When evacuation is instructed* -> `next24`
   - *During a cyclone* -> `final12`
   - *Post-cyclone measures* / *Recover and build* -> `after`
4. Set `_coastal_only: true` only if the sentence literally refers to the coast/beach/sea/surge.
5. Run `python scripts/build_checklist.py` — it validates the file, rebuilds `checklist_rules.json`, and copies both into `app/static/data/`. No code change is needed.
