# Orchestration Assets

Reusable prompts and workflow patterns for multi-agent planning work on this
repository: research fan-outs, mixture-of-experts proposal debates, and
adversarial pressure-testing of candidate roadmap items.

These assets exist so that an orchestrating agent (human-driven or autonomous)
can convene a high-quality expert panel about this codebase without
reconstructing the framing from scratch each time. They encode what we have
learned about getting useful work out of subagents on this project:

- **Experts propose; skeptics refute; the orchestrator verifies.** No proposal
  survives on authority. Every claim that touches GEPA/DSPy internals or this
  repo's seams gets checked against the actual source before it is presented.
- **Ground every prompt in the repo's current state.** The panel prompts below
  take a context block (architecture summary, open problems, recent findings).
  Stale context produces confident nonsense; the orchestrator owns refreshing it.
- **Signal before mechanism.** This project's hardest lesson: selection/gating
  machinery is worthless when the eval suite has no discriminating signal.
  Panelists are instructed to ask "where does the signal come from?" first.

## Layout

```
orchestration/
├── README.md                  # this file
├── prompts/
│   ├── panel/                 # mixture-of-experts debate roles
│   │   ├── _shared_preamble.md
│   │   ├── evolutionary_computation_expert.md
│   │   ├── ml_systems_engineer.md
│   │   ├── statistics_eval_expert.md
│   │   ├── agent_frameworks_expert.md
│   │   └── red_team_skeptic.md
│   └── research/
│       └── literature_scout.md
└── workflows/
    └── proposal_debate.md     # how to wire the panel into a Workflow run
```

## Usage

1. Build a **context pack**: current architecture one-pager, the open-problems
   list, and any fresh research findings. Keep it under ~2k words.
2. Spawn one agent per panel prompt, substituting `{{CONTEXT_PACK}}` and
   `{{QUESTION}}`. Run panelists in parallel; they must not see each other's
   output in the first round.
3. Feed all first-round proposals to the red-team skeptic. The skeptic's job is
   to kill, not to polish.
4. The orchestrator pressure-tests every surviving claim against the codebase
   (file:line) before synthesis. Claims that fail verification are dropped or
   downgraded, never silently repaired.
