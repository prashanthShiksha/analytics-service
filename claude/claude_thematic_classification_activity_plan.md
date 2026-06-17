# Thematic Classification Activity — Plan

> Renames `thematic_analysis_activity` → `thematic_classification_activity`.
> Companion files: `seed_prompts.sql`, `env.example`.

## 1. Scope and Source Material

This plan replaces the current thematic analysis flow with a content-quality-gated classification pipeline. It is driven by:

- `prompt_version.csv` (prompts: `theme_classification`, `story_rating`, `pii_detection`) — restructured into `system_prompt` / `user_prompt` pairs in `seed_prompts.sql`.
- The `analysis_results` table (existing columns: `id`, `submission_id`, `tenant_code`, `theme_id`, `analysis_type`, `statements`, `statement_type`, `improvement_environment`, `confidence_score`, `justification`, `multi_theme_mapped`, `meta_data`), extended per Section 3 below.
- New configuration in `.env` / `env.example` (Section 4).

## 2. Schema Changes

Add to `analysis_results`:

| Column | Type | Purpose |
|---|---|---|
| `content_quality` | TEXT | One of `Standard`, `Others`, `Unknown/Unclear`, `Flagged`. Set exactly once per processed statement. |
| `similarity_score` | FLOAT | Cosine similarity score from the local embedding match against the best approved theme. NULL if the statement never reached local classification (e.g. flagged/unknown). |

`confidence_score` already exists and is reused for the LLM-fallback path (Step 8) rather than adding a new column.

`theme_id` is only populated when `content_quality = 'Standard'`. For `Others`, `Unknown/Unclear`, and `Flagged`, `theme_id` stays NULL — this is the "without mapping/tagging to theme" requirement from the spec.

Also extend `prompts` / `prompt_version` with `system_prompt` and `user_prompt` TEXT columns (replacing or supplementing the single `content` column — see Open Decision 1).

## 3. Pipeline Flow

```
For each submission, for each configured submission_type:
  │
  ├─ 1. Based on the submission_type, process either PROCESS_CONFIG_STORY or
  │      PROCESS_CONFIG_DISCUSSION (see Section 3b — both now include a
  │      thematic_analysis step, so story and discussion columns both reach
  │      full theme classification).
  │
  ├─ 1b. If submission_type is in the "discussion" group (e.g. challenges):
  │      split on THEMATIC_STATEMENT_DELIMITER ("|") into N separate statements,
  │      process each as its own unit through steps 2-9. See Section 3a.
  │
  ├─ 2. Word-count / garbage check
  │      ├─ word_count < MINIMUM_THEME_WORD_COUNT, OR matches garbage/spam pattern
  │      │     → content_quality = 'Unknown/Unclear', theme_id = NULL, STOP (step 4)
  │      └─ Otherwise → continue
  │
  ├─ 3. Safety check (no LLM) — keyword blocklist + moderation library
  │      ├─ Flagged?  → content_quality = 'Flagged', theme_id = NULL, STOP (step 4)
  │      └─ Not flagged → continue
  │
  ├─ 4. [STOP point reached above] — no further processing for Unknown / Flagged.
  │      NOTE on priority: Step 2 runs before Step 3, so if a statement is BOTH
  │      too short/garbled AND would otherwise trip the safety check, the
  │      word-count gate wins and it is marked 'Unknown/Unclear', not 'Flagged'.
  │      This is intentional (confirmed) — the safety check never even runs
  │      once Step 2 has already stopped the pipeline for that statement.
  │
  ├─ 5. Fetch approved themes from `themes` table (id, name, definition)
  │
  ├─ 6. Local classification (no LLM):
  │      embed statement with all-MiniLM-L6-v2
  │      compare against embeddings of all approved themes (compute the
  │      similarity against the name embedding and the definition embedding
  │      separately for each theme, then take the best of those two per theme)
  │      take BEST match across all themes → similarity_score = best cosine similarity
  │      ├─ similarity_score >= SIMILARITY_SCORE_THRESHOLD (0.65)
  │      │     → assign theme_id = best match, content_quality = 'Standard', STOP
  │      └─ similarity_score < threshold → continue to step 7
  │
  ├─ 7. Build LLM prompt: fetch latest theme_classification prompt version,
  │      substitute {{approved_themes}} with full theme list (name + definition),
  │      substitute statement placeholder with the original statement text
  │
  ├─ 8. Call LLM, parse classification_confidence_score (0-10 scale)
  │      ├─ confidence_score >= LLM_CONFIDENCE_SCORE_THRESHOLD (8)
  │      │     → assign theme_id = LLM's theme, confidence_score = score
  │      └─ confidence_score < threshold → theme_id stays NULL
  │
  └─ 9. Finalize content_quality:
         ├─ similarity_score >= 0.65  OR  confidence_score >= 8
         │     → content_quality = 'Standard', theme_id = matched theme
         └─ otherwise (understandable but vague / multi-theme / off-taxonomy /
            low confidence)
               → content_quality = 'Others', theme_id = NULL
```

### 3a. Discussion-specific splitting logic (placeholder block)

Per the spec, discussion-type columns (challenges) need their own handling because a single cell can contain multiple `|`-delimited statements that must each be run through the full pipeline independently, with their own row in `analysis_results`.

```python
def process_discussion_statement_type(raw_cell_value: str, submission_id, tenant_code):
    """
    Split a discussion-style cell into individual statements and process
    each one through the standard pipeline (steps 2-9), writing one
    analysis_results row per resulting statement.

    Reserved space for discussion-specific logic that does NOT apply to
    objective/solutions statement types — e.g.:
      - de-duplication of near-identical split fragments
      - minimum fragment length re-check (a split could produce a fragment
        that's too short even if the original combined statement passed
        the word-count gate)
      - re-merging fragments that are clearly continuations
        (e.g. ".|and also" type artifacts from inconsistent data entry)
      - tracking the *original* unsplit statement somewhere (meta_data?)
        so we can still answer "what was the raw discussion text" for audit

    # TODO: fill in actual splitting/cleanup rules once we've looked at
    # real discussion-column data and seen what split artifacts look like.
    """
    statements = [s.strip() for s in raw_cell_value.split(
        os.environ["THEMATIC_STATEMENT_DELIMITER"]
    ) if s.strip()]

    for statement in statements:
        run_classification_pipeline(
            statement=statement,
            submission_id=submission_id,
            tenant_code=tenant_code,
            statement_type="challenges",
            analysis_type="theme",
        )
        # each call inserts its own analysis_results row;
        # multi_theme_mapped logic from the LLM prompt still applies
        # per-fragment, not across fragments
```

### 3b. Resolving submission_type → process steps via existing config

These two config keys already exist in `.env` and are the actual source of truth for "what runs for this submission_type" — they supersede the earlier `THEMATIC_STATEMENT_TYPES` idea from Section 1/4 below, which can be dropped:

```python
PROCESS_CONFIG_STORY: str = Field(
    default='[{"name": "pii_detection", "columns": ["objective"]}, '
            '{"name": "thematic_analysis", "columns": ["objective"]}]'
)
PROCESS_CONFIG_DISCUSSION: str = Field(
    default='[{"name": "pii_detection", "columns": ["challenges"]}, '
            '{"name": "thematic_analysis", "columns": ["challenges"]}]'
)
```

Resolved: `PROCESS_CONFIG_DISCUSSION` now includes a `thematic_analysis` entry alongside `pii_detection`, matching `PROCESS_CONFIG_STORY`'s shape. This confirms discussion (`challenges`) statements go through the full Steps 2-9 pipeline, not just PII detection — consistent with Step 1b and Section 3a.

Step 1 reads the appropriate config (`PROCESS_CONFIG_STORY` for story-type submissions, `PROCESS_CONFIG_DISCUSSION` for discussion-type) and, for each `{"name": ..., "columns": [...]}` entry, dispatches to the matching process (`pii_detection` or `thematic_analysis`/`thematic_classification`) against the listed column(s). The `thematic_analysis` entry is what triggers Steps 2-9 for that submission_type's columns; `pii_detection` is a separate, already-existing process not covered by this plan.

Implication for naming: since the config key is literally `"thematic_analysis"` (not `"thematic_classification"`), either the config value string needs updating to `"thematic_classification"` as part of the rename in Section 8, or the activity code keeps matching on the string `"thematic_analysis"` while the *job/entrypoint* itself is renamed. Worth deciding which before the final rename step — flagged in Open Decisions below.

## 4. Configuration (`.env` / `env.example`)

All new keys, with defaults, are in `env.example` (already created alongside this plan):

- `PROCESS_CONFIG_STORY` / `PROCESS_CONFIG_DISCUSSION` — **already exist in `.env`**, now the actual source of truth for which columns get `thematic_analysis` applied (see Section 3b). `THEMATIC_STATEMENT_TYPES` from the original draft of this plan is superseded by these and should be removed from `env.example` rather than added — see Section 8 step list, updated.
- `THEMATIC_STATEMENT_DELIMITER` — split character for discussion-type cells (`|`)
- `MINIMUM_THEME_WORD_COUNT` — gate for Step 2
- `EMBEDDING_MODEL_NAME` — `all-MiniLM-L6-v2`
- `SIMILARITY_SCORE_THRESHOLD` — `0.65`, gate for Step 6
- `LLM_CONFIDENCE_SCORE_THRESHOLD` — `8`, gate for Step 8
- `SAFETY_KEYWORD_BLOCKLIST_PATH` / `SAFETY_MODERATION_LIBRARY` — Step 3 mechanism
- `THEMATIC_DISCUSSION_BATCH_SIZE` — batching for Step 10/3a

I have not read your actual `.env` (per your instruction) — `env.example` was built from the spec's requirements only, so cross-check the threshold defaults against whatever you're already running. Note `env.example` as currently written still includes `THEMATIC_STATEMENT_TYPES`; flagging it for removal here rather than silently deleting it, since other code may already reference it.

## 5. Safety Flagging Mechanism (Step 3, no LLM)

Two layers, both non-LLM, run after the word-count gate (Step 2) has already passed:

1. **Keyword/regex blocklist** — fast, deterministic, catches explicit abusive language and obvious PII patterns (phone numbers, email addresses, ID numbers) via regex. Cheap and auditable, but brittle to obfuscation/spelling variants.
2. **Third-party moderation library** — catches what the blocklist misses (e.g. `better-profanity` for profanity scoring, `presidio` or similar for structured PII entity detection like names/addresses). Adds recall at the cost of an extra dependency and runtime.

A statement is `Flagged` if **either** layer trips. Order matters for performance: run the cheap regex/keyword pass first; only run the heavier moderation library if the cheap pass doesn't already flag it (short-circuit, don't always run both).

Exact library choice (`SAFETY_MODERATION_LIBRARY` in `env.example`) is left as a config value rather than hardcoded — swap it without touching pipeline logic. Note this only runs for statements that already passed the Step 2 word-count gate (see Step 4 priority note in Section 3).

## 6. Open Decisions (need your call before implementation)

1. **system_prompt/user_prompt split semantics.** The existing `prompt_version.content` is one blob per prompt. I split each into `system_prompt` (role, guidelines, output format, rules) and `user_prompt` (the actual "now classify this" instruction + placeholders) in `seed_prompts.sql`. This is a reasonable default but is my judgment call, not something the CSV specified — please review the split for `theme_classification` in particular, since it's the one driving runtime behavior.

2. **Placeholder naming mismatch.** The CSV's existing theme prompt uses `{{statements}}` (plural — designed for batch classification of a list in one LLM call). Your spec's Step 7 says replace `{{statement}}` (singular — implying one LLM call per statement). I kept `{{statements}}` in `seed_prompts.sql` since that's literally what's in your source CSV, but the pipeline pseudocode in Section 3 assumes one-call-per-statement per your spec. **These two don't currently agree.** Pick one:
   - **Batch mode**: collect all statements that fell through to LLM classification (Step 7) within a submission/run, and pass them as a list in one call — cheaper, matches the existing prompt's actual design and its multi-object JSON output format.
   - **Per-statement mode**: one LLM call per statement — matches your spec literally, simpler to reason about and retry, but more LLM calls and the existing prompt's output format (an array, designed for batches) would be overkill for a single statement.
   I'd lean toward batch mode given the prompt was clearly designed for it, but this changes the calling code shape, so flagging rather than deciding silently.

3. **Discussion splitting logic (Section 3a) is a stub.** I left explicit `# TODO` space per your instruction ("have some space to fill this logic") rather than guessing cleanup rules without seeing real discussion-column data. Once you can share a handful of real `challenges` cell values (even just 5-10 anonymized examples), I can fill in the actual split/cleanup rules instead of leaving them as placeholders.

4. **Garbage/spam detection in Step 2.** The spec gives `"test test"` as an example but the actual detection rule beyond word-count isn't fully specified (e.g. do we need repeated-token detection, or just the word-count floor?). Current plan only implements the word-count gate; flag if you want pattern-based garbled-text detection too (e.g. repeated single tokens, no vowels, etc.).

5. **`prompts`/`prompt_version` column migration.** `seed_prompts.sql` assumes `content` is being replaced by `system_prompt` + `user_prompt`. If `content` needs to stay (e.g. other code still reads it), the seed needs an extra `content = system_prompt || user_prompt` assignment — let me know if that's the case before running this against a real DB.

6. **`thematic_analysis` vs `thematic_classification` naming in `PROCESS_CONFIG_*`.** Per Section 3b, the existing config dicts use the literal string `"thematic_analysis"` as the process name. The overall rename in this plan is `thematic_analysis_activity` → `thematic_classification_activity`, but that's the job/entrypoint name, not necessarily this config string. Decide: (a) update the config value to `"thematic_classification"` and have the dispatcher match on the new name, or (b) leave the config string as `"thematic_analysis"` and have it map to the new classification logic internally, keeping the external config vocabulary stable. Either works; (b) is lower-risk if anything else reads this config today.

## 7. Files in This Change

- `seed_prompts.sql` — seeds `prompts` + `prompt_version` with the system/user split for all 3 existing prompts (theme_classification, story_rating, pii_detection). `{{approved_themes}}` left as a placeholder, not inlined.
- `env.example` — all new config keys with comments (no real secrets; `.env` itself was not read or modified).
- This plan.

## 8. Suggested Implementation Order

1. Run the `analysis_results` migration (add `content_quality`, `similarity_score`).
2. Run the `prompts`/`prompt_version` migration (add `system_prompt`, `user_prompt`) — resolve Open Decision 5 first.
3. Apply `seed_prompts.sql`.
4. Wire up Step 2 (word count / garbage gate) and Step 3 (safety) in that order — both are pure-Python, no LLM/embedding dependency, fastest to test in isolation. Confirm the word-count-wins priority (Section 3, Step 4 note) is reflected in tests.
5. Wire up Step 6 (local embedding classification, name + definition embeddings) against a small known theme set; validate similarity scores look sane before touching the LLM fallback.
6. Wire up Step 7-8 (LLM fallback), resolving Open Decision 2 (batch vs per-statement) first since it determines the function signature.
7. Wire up Section 3a discussion splitting once Open Decision 3 is resolved with real example data.
8. Resolve Open Decision 6 (`thematic_analysis` vs `thematic_classification` string in `PROCESS_CONFIG_*`), then rename the activity entrypoint/job from `thematic_analysis_activity` to `thematic_classification_activity` last, once the new logic is verified end-to-end under the old name — reduces risk of partially-migrated state if something needs to roll back mid-implementation.