Evolved `all-tools` with Phase 2 (tool descriptions).

## Scores

| Split | Before | After | Change | Notes |
|---|---:|---:|---:|---|
| val | 1.000 | 1.000 | +0.000 | 4 examples, 16.7% chance across 6 options |
| holdout | 1.000 | 1.000 | +0.000 | 8 examples, never optimized against |

## Evidence

cross-tool accepted: overall 100.0% -> 100.0% (+0.0%); no tool regressed beyond a 0.0% tolerance (0 improved, 2 not measurable)

## Gates

- ✓ cross_tool: no tool regressed beyond a 0.0% tolerance (0 improved, 2 not measurable)
- ○ tblite: benchmark 'tblite' not found in /tmp/hasereview/r5 (set HERMES_BENCH_TBLITE to point at it)

## Run

- Optimizer: GEPA, 2 iteration(s)
- Eval dataset: /tmp/hasereview/e/ds - 12 train / 4 val / 8 holdout
- Cost: no model calls recorded
- Reflection model: openai/gpt-4.1
- Eval model: openai/gpt-4.1-mini
- Descriptions changed: 1
- Description size: 1,446 to 965 chars (-481)
- Optimization wall clock: 0.0s
- Run artifacts: output/tools/20260731_030412

## Diff

```diff
diff --git a/tools/shell_tools.py b/tools/shell_tools.py
index 7f24d86..e47c4bc 100644
--- a/tools/shell_tools.py
+++ b/tools/shell_tools.py
@@ -4,7 +4,7 @@ from tools import registry
 
 TERMINAL_SCHEMA = {
     "name": "terminal",
-    "description": "Run a shell command in a persistent session. Prefer the purpose-built file tools over shell equivalents wherever one exists. Prefer the purpose-built file tools over shell equivalents wherever one exists. Prefer the purpose-built file tools over shell equivalents wherever one exists. Prefer the purpose-built file tools over shell equivalents wherever one exists. Prefer the purpose-built file tools over shell equivalents wherever one exists. Prefer the purpose-built file tools over shell equivalents wherever one exists. ",
+    "description": 'Run a shell command in a persistent session.',
     "parameters": {
         "type": "object",
         "properties": {
```
