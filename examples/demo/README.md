# Demo failures for the one-command loop

`failures.jsonl` contains three secret-clean failed attempt traces.

Use it after registering a repo target that has `skill:github-code-review`:

```bash
.venv/bin/hermes-evolve --root .evolution-state loop once \
  --target skill:github-code-review \
  --trace-path examples/demo/failures.jsonl \
  --strategy deterministic \
  --scoring-strategy deterministic-rubric \
  --export-out .evolution-review
```

The file is intentionally tiny: enough for train/val/holdout smoke testing, not a real benchmark. Because apparently demos should be reproducible instead of spiritually accurate.
