# Python review benchmark (KAFO-EVO-001, gate 2)

Small, versioned set of Python snippets with **seeded, known defects** and
**clean controls**, for scoring code-review output against ground-truth
defect IDs.

> **Synthetic data.** Every snippet was written by hand for this project as a
> short, self-contained example. Defects were seeded on purpose; nothing was
> copied from public benchmarks or real codebases. A good score here is **not**
> evidence that Hermes improves on real code review.

## Version v1

| Split | File | Cases | Clean | Defects |
|---|---|---|---|---|
| train | `v1/train.jsonl` | 28 | 4 | 31 |
| validation | `v1/validation.jsonl` | 10 | 2 | 10 |
| test (sealed) | `v1/sealed/test.jsonl` | 10 | 2 | 9 |

Difficulty mix: 19 simple, 17 moderate, 12 complex. Every split contains all
three difficulties and at least one clean control.

## Format

JSONL, one case per line, validated by [`schema.json`](schema.json):

```json
{"case_id": "pyrev-001", "split": "train", "language": "python",
 "code": "def find_user(conn, username):\n    ...",
 "context": {"difficulty": "simple", "domain": "database"},
 "defects": [{"defect_id": "pyrev-001-d1", "category": "security",
              "line_start": 2, "line_end": 3, "severity": "critical",
              "description": "..."}]}
```

- `line_start` / `line_end` are 1-based and inclusive, counted on `\n` in `code`.
- `defect_id` is `<case_id>-d<n>` and unique across all splits.
- `defects` is empty for clean control cases.
- `category` is a free string in the schema; v1 uses only: `security`,
  `logic_error`, `error_handling`, `input_validation`, `resource_management`,
  `concurrency`, `performance`, `api_misuse`
  (`evolution.adaptive.benchmark.DEFECT_CATEGORIES`).
- A case labels every defect a careful reviewer should report. Clean controls
  are meant to contain no reportable defect; findings on them count as false
  positives.

## Sealed test split and integrity

- The test split lives in its own file under `v1/sealed/`.
- `v1/MANIFEST.json` records the SHA-256 of `schema.json` and of every split
  file. The SHA-256 of the manifest itself is pinned in
  `evolution/adaptive/benchmark.py` (`PINNED_MANIFEST_SHA256`), so any edit to
  cases, labels or schema fails loading until the pin is changed in reviewed code.
- `.gitattributes` disables line-ending conversion so hashes match on every OS.

## Loading

```python
from evolution.adaptive.benchmark import load_benchmark, load_split

open_splits = load_benchmark()          # train + validation only
train = load_split("train")             # tuple of frozen BenchmarkCase objects
load_split("test")                      # raises SealedSplitError
load_split("test", allow_sealed_test=True)  # final evaluation only
```

Loading train or validation never opens the sealed file.

During optimisation or selection, wrap the work in
`evolution.adaptive.benchmark.optimisation_phase()` (the adaptive loop already
does). Inside it the test split cannot be loaded, and opening any file under
`python_review/.../sealed/` raises `SealedSplitError`. Score reviews with
`evolution.adaptive.evaluation.evaluate_open`. The one-off final evaluation is
`evaluate_sealed_test(reviews, final_evaluation=True)`, run outside optimisation.

Read-only checks:

```bash
python -m evolution.adaptive.benchmark verify     # hashes all files, validates open splits
python -m evolution.adaptive.benchmark manifest   # prints a recomputed manifest; writes nothing
```

## Changing the benchmark

Do not edit a released version in place. Create `v2/`, write its manifest
from the `manifest` command output, and add its pin in reviewed code. Agents
under optimisation must never modify these files.
