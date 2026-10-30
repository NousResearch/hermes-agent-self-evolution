"""Synthetic fixtures for content-retention tests.

These are deliberately artificial so the test suite carries no real-world
content: they exercise the *shape* of the problem (a skill that contains
runnable commands and guard lines, and an "evolved" variant that drops
them) without embedding anyone's actual playbook.
"""

BASELINE = """---
name: fixture-skill
description: A synthetic fixture used by the content-retention unit tests.
---

# Fixture Skill

## Setup

Run the following to prepare the environment:

```bash
docker compose up -d
kubectl get pods -n demo
```

## Rules

- Never run `rm -rf` against a production namespace.
- Always check `kubectl describe pod` before restarting a workload.
- You must run the smoke test before declaring the rollout complete.

## Commands

```bash
pytest tests/ -q
git log --oneline -8
curl -s https://example.invalid/health
```
"""

# Simulates the failure mode described in issue #200: the commands and the
# guard lines are removed, the prose is kept, and the result is shorter.
EVOLVED_GUTTED = """---
name: fixture-skill
description: A synthetic fixture used by the content-retention unit tests.
---

# Fixture Skill

## Setup

Prepare the environment with the standard container workflow.

## Rules

Follow the operational guidelines carefully and consider the relevant risks.

## Commands

Use the normal tooling for verification.
"""

# A legitimate rewrite: same commands, same guard lines, different wording.
EVOLVED_EQUIVALENT = """---
name: fixture-skill
description: A synthetic fixture used by the content-retention unit tests.
---

# Fixture Skill

## Setup

Bring the environment up first:

```bash
docker compose up -d
kubectl get pods -n demo
```

## Rules

- Do not run `rm -rf` against a production namespace.
- Always inspect `kubectl describe pod` before restarting a workload.
- You have to run the smoke test before declaring the rollout complete.

## Commands

```bash
pytest tests/ -q
git log --oneline -8
curl -s https://example.invalid/health
```
"""

# A pure addition: baseline plus extra content.
EVOLVED_GROWTH = BASELINE + """
## Troubleshooting

If the pod is pending, inspect the events:

```bash
kubectl describe pod -n demo
```
"""
