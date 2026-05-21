# Hermes PHI Surface Map

> **Status**: Living document — update as new surfaces are discovered.
> **Epic**: bd-jqrzp (Hermes PHI Redaction Plugin and Safety Integration)
> **Created**: 2026-05-21

## Overview

This document maps all surfaces where Protected Health Information (PHI) may enter,
traverse, or leave the Hermes Agent process. For each surface we identify the
earliest safe interception point, whether Hermes built-in privacy controls cover it,
and whether the new Hermes PHI plugin must intervene.

### Surface Map Quick Reference

| # | Surface | Data Shape | PHI Possible | Earliest Intercept | Built-in Coverage | Plugin Action |
|---|---------|-----------|-------------|-------------------|------------------|--------------|
| 1 | User CLI text input | `str` (stdin/CLI arg) | Yes | Input parsing layer | None | Redact before context assembly |
| 2 | Document/file ingestion | `str` (read from path/stdin) | Yes | File read wrapper | None | Redact on read |
| 3 | Tool output capture | `str` (stdout/stderr) | Yes | Tool execution result handler | None | Redact before context append |
| 4 | Skill/tool prompt load | SKILL.md frontmatter/body | Low (may contain synthetic examples) | Skill loader | None | Review only |
| 5 | Session history replay | `~/.hermes/sessions/*.json` | Yes (if sessions contain PHI) | Session importer | Hermes `privacy.redact_pii` (session export only) | Redact on import |
| 6 | Context/prompt assembly | `str` (concatenated messages) | Yes | Pre-send LLM call | None | **BLOCK or redact before model** |
| 7 | Remote LLM provider call | API request body | Yes | Just before `client.chat.completions.create(...)` | None | **Redact payload; fail-closed on unavailability** |
| 8 | Memory/state persistence | `str` (checkpoint/save) | Yes | Before write to disk | None | Redact before write |
| 9 | Audit/eval dataset export | JSONL examples | Yes (if sessions mined) | Dataset builder export | Secret detection (API keys only, not PHI) | Redact before export |
| 10 | Logs/traces | `str` (log lines) | Yes | Before logger call | None | Redact before emit |

---

## Surface 1: User CLI Text Input

**Path**: Hermes Agent CLI entry point → `input()` or CLI argument parser.

**Data shape**: `str` — the user's raw natural-language request.

**PHI risk**: **High**. Users may paste clinical text, patient identifiers, or
referral notes directly into the CLI.

**File/function**: In the Hermes Agent process boundary. For the self-evolution
toolchain, user text enters via:
- `evolution/core/external_importers.py` → `ClaudeCodeImporter`, `CopilotImporter`,
  `HermesSessionImporter` — all read pre-existing session data, not live input.

**Earliest safe interception point**: Input parsing layer before any context
assembly. In Hermes Agent, this is the REPL loop or request handler.

**Built-in coverage**: Hermes Agent has a `privacy.redact_pii` setting documented
for session export. It is **not guaranteed** to fire on live user input.

**Plugin action**: **Redact** before assembling context for LLM.

---

## Surface 2: Document / File Ingestion

**Path**: `cat file.txt | hermes` or `hermes --file document.txt` or via tools.

**Data shape**: `str` — raw file content up to the model's context window.

**PHI risk**: **High**. Clinical notes, discharge summaries, referrals all contain
PHI as a matter of course.

**Earliest safe interception point**: At the file-read boundary, before content
enters any processing pipeline.

**Built-in coverage**: None known.

**Plugin action**: **Redact** before context assembly.

---

## Surface 3: Tool Output Capture

**Path**: Tool execution → stdout/stderr capture → appended to conversation context.

**Data shape**: `str` — tool/function return values and side-channel output.

**PHI risk**: **Medium-High**. A tool that queries a medical database, parses a
clinical document, or retrieves patient records may return PHI in its output.

**File/function**: In Hermes Agent's tool execution loop, the result handler that
appends tool output to the message list.

**Earliest safe interception point**: Before tool output is appended to the
conversation history.

**Built-in coverage**: None known.

**Plugin action**: **Redact** before appending to context. For sensitive tools,
**BLOCK** if PHI confidence exceeds configured threshold.

---

## Surface 4: Skill / Tool Prompt Load

**Path**: SKILL.md loading → system prompt assembly.

**Data shape**: `str` — SKILL.md frontmatter + markdown body. In the
self-evolution repo, skills are loaded via `skill_module.load_skill()` and
optimized by GEPA.

**PHI risk**: **Low**. Skill files should not contain PHI as they are
instructions, not patient data. However, synthetic eval examples _could_
inadvertently include PHI-like data if the LLM generating them produces
realistic-looking identifiers.

**Earliest safe interception point**: Not applicable for production — skills are
prompts, not payloads. Review only.

**Built-in coverage**: None needed.

**Plugin action**: **No intervention** for normal usage. Review synthetic
examples for accidental PHI leakage.

---

## Surface 5: Session History Replay

**Path**: `~/.hermes/sessions/*.json` → `HermesSessionImporter.extract_messages()`
→ `build_dataset_from_external()`.

**Data shape**: JSON with user messages, assistant responses, tool calls/results.

**PHI risk**: **High**. Sessions may contain the entire conversation history
including PHI the user discussed with the agent.

**File/function**:
- `evolution/core/external_importers.py` → `HermesSessionImporter` (lines ~330-390)
  reads session JSON files
- `SECRET_PATTERNS` regex only covers API keys/tokens — **does not cover PHI**

**Earliest safe interception point**: At the `extract_messages()` return boundary,
before messages enter the relevance filter and dataset builder.

**Built-in coverage**: Hermes Agent has `privacy.redact_pii` for session export,
but the self-evolution importer reads session data outside that mechanism.
The existing `SECRET_PATTERNS` covers credentials only — not PHI.

**Plugin action**: **Redact** session messages on import. **Block** import of
sessions that cannot be fully redacted.

---

## Surface 6: Context / Prompt Assembly

**Path**: Skill text + conversation history + tool outputs → assembled `str` or
message list → model-bound payload.

**Data shape**: `str | list[LLMMessage]` — the fully assembled context passed to
the LLM.

**PHI risk**: **Highest**. This is the aggregation point for all prior surfaces.
Any unredacted PHI from user input, documents, tools, or session history converges
here and is about to leave the process.

**File/function**: In Hermes Agent, the function that prepares the LLM API request
body. In the DSPy/GEPA pipeline, the `TaskWithSkill` signature creates the prompt
with `skill_instructions` and `task_input`.

**Earliest safe interception point**: **Immediately before the LLM API call**.
This is the last-chance barrier.

**Built-in coverage**: None.

**Plugin action**: **Evaluate with Safety Kernel**. If verdict is **ALLOW** with
redactions — replace payload with redacted version. If **BLOCK** — suppress the
payload entirely and return a failure indicator.

---

## Surface 7: Remote LLM Provider Call

**Path**: `client.chat.completions.create(...)` or equivalent DSPy LM call.

**Data shape**: API request body with model, messages, and parameters.

**PHI risk**: **Highest**. Raw payload leaves the process boundary to a third-party
API endpoint. This is the primary exfiltration risk.

**File/function**: In the self-evolution repo, DSPy LM calls are made through
`dspy.LM(model)` which wraps OpenAI-compatible API endpoints. In Hermes Agent,
through the configured LLM client.

**Earliest safe interception point**: Before the HTTP request is dispatched.

**Built-in coverage**: None.

**Plugin action**: **Redact payload; fail-closed if Safety Kernel unavailable.**
The redacted payload from Surface 6 is what gets sent. If redaction fails or
evaluator is unavailable, the call must be suppressed.

---

## Surface 8: Memory / State Persistence

**Path**: Agent state → serialized JSON → disk or database.

**Data shape**: JSON with conversation history, agent state, and metadata.

**PHI risk**: **High**. If the agent persists its state between sessions, PHI
from the conversation will be included.

**File/function**: Depends on the persistence mechanism (file-based checkpoints,
database records). For the self-evolution toolchain, session importers write
JSONL dataset files.

**Earliest safe interception point**: Before serialization to disk.

**Built-in coverage**: None known.

**Plugin action**: **Redact** before any write to durable storage.

---

## Surface 9: Audit / Eval Dataset Export

**Path**: Dataset builder → `EvalDataset.save()` → JSONL files on disk.

**Data shape**: `EvalExample` dicts with `task_input` and `expected_behavior`.

**PHI risk**: **Medium-High**. If source sessions contain PHI, exported datasets
will inherit it. These files may be committed to version control, shared, or
used for future optimization runs.

**File/function**:
- `evolution/core/dataset_builder.py` → `EvalDataset.save()`
- `evolution/core/external_importers.py` → `build_dataset_from_external()`

**Earliest safe interception point**: Inside `EvalExample.to_dict()` or at the
`save()` boundary.

**Built-in coverage**: The existing `_contains_secret()` check in
`external_importers.py` scans for API keys/tokens — **does not scan for PHI**.

**Plugin action**: **Redact** `task_input` and `expected_behavior` before
serialization. Must not block dataset creation if redaction succeeds.

---

## Surface 10: Logs / Traces

**Path**: Python `logging` calls, `rich.console` output, stdout/stderr during
development.

**Data shape**: `str` — log messages that may contain user input, tool output,
or assembled context.

**PHI risk**: **Medium**. Development logs, debug traces, and error messages may
inadvertently include PHI from the payloads being processed.

**File/function**: All modules. In the self-evolution codebase, `rich.console`
is used extensively for user-facing output.

**Earliest safe interception point**: At the logger/console boundary. Hard to
retrofit — requires structured logging with PHI-safe formatters.

**Built-in coverage**: None.

**Plugin action**: **Redact** via a logging filter/formatter where feasible.
Document as a known gap for now — full log redaction requires a broader
infrastructure change.

---

## Uncovered Surfaces & Known Gaps

1. **Log redaction**: Log/trace output is the hardest surface to retrofit.
   Immediate priority is payload surfaces (6, 7). Log redaction should be
   addressed in a follow-up.

2. **Error messages**: Exception messages that include PHI context leak through
   error reporting. Requires structured exception handling.

3. **CLI output**: Hermes Agent's console output may render PHI from the
   response. Redaction of the model response itself (Surface 6) is the
   primary defense.

4. **DSPy LM call interception**: The self-evolution toolchain uses `dspy.LM()`
   for LLM calls. This is an indirect API — intercepting it would require
   wrapping or monkey-patching DSPy's LM call method. For direct Hermes Agent
   integration, the LLM client is the correct interception point.

5. **Plugin insertion point**: The Hermes plugin system does not exist yet in
   the self-evolution repo. The `hermes_phi.plugin.HermesPHIPlugin` created
   alongside this document assumes a wrapping/adapter pattern where the calling
   code explicitly invokes the plugin before LLM calls.

---

## Recommended Implementation Order

```
bd-jqrzp.1 (this doc)
    └── Identifies surfaces and intercept points
    └── Informs plugin API requirements

bd-jqrzp.2 (hook spike)
    └── Proves Safety Kernel can be called from Hermes context
    └── Validates surface 6 / 7 interception

bd-jqrzp.3 (production plugin)
    └── HermesPHIPlugin implementing coverage for surfaces 1-9
    └── Fail-closed on Safety Kernel unavailability
    └── Comprehensive test coverage
```

## References

- `hermes_phi/plugin.py` — HermesPHIPlugin implementation
- `hermes_phi/plugin_proof.py` — Hook spike / adapter proof
- `tests/test_hermes_phi_plugin.py` — Test coverage
- `llm-common` Safety Kernel: `llm_common/core/kernel.py`, `llm_common/core/safety.py`,
  `llm_common/core/detector.py`, `llm_common/core/policy.py`, `llm_common/core/audit.py`
- Merged at commit `396146389e9e5017cee881bea349c41e7a1f593d`
- PR: https://github.com/stars-end/llm-common/pull/114

