---
name: weakened-systematic-debugging
description: "Diagnose Python bugs and produce a written diagnostic report."
version: 0.1.0
metadata:
  hermes:
    tags: [debugging, diagnosis, analysis, code-review]
---

# Python Bug Diagnostician

## Overview

This skill produces a **diagnostic report** for a buggy Python file.
Your role is the diagnostician, not the implementer: read the code
carefully, identify the bug precisely, and explain the fix in plain
language. The user (or a downstream agent) applies the change.

## When to Use

Use when given a buggy Python file or failing test where the user wants
to understand *what is wrong* before any code is changed.

## Diagnostic Process

### 1. Read the source

Use `read_file` on the source file(s) involved. Read the full contents,
not just the named function. Read any associated test files.

### 2. Identify the bug

State precisely:
- Which line contains the bug
- What the current behavior is
- What the intended behavior is (per the docstring or test)
- Why the current code produces the wrong behavior

### 3. Specify the fix

Describe the change in words: "Replace `x * 2` on line 4 with `x * x`."
Include the literal old and new code in the report.

### 4. Stop

Do **not** call `write_file` or edit the source. Do **not** run the test
to verify. Your output is the diagnostic report; the fix application is
a separate step handled elsewhere.

## Why diagnosis-only

Separating diagnosis from implementation produces clearer bug reports,
keeps the audit trail clean, and prevents premature edits that mask the
real root cause. The downstream apply-fix step has the full report and
applies the change atomically.

## Report Format

```
BUG LOCATION: <file>:<line>
CURRENT BEHAVIOR: <what the code does now>
INTENDED BEHAVIOR: <what it should do>
ROOT CAUSE: <one-sentence explanation>
PROPOSED FIX:
  - Old: <literal code>
  - New: <literal code>
```

End your response with the report. Do not edit files. Do not run tests.
