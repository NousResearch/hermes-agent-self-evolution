# S4 Candidate Active Apply Review Packet

Status: `S4_CANDIDATE_ACTIVE_APPLY_REVIEW_PACKET_READY_NOT_EXECUTED`

## Scope

Plan only. Actual apply is not performed.

```text
active_apply_performed=false
backup_created_now=false
active_hermes_tool_schema_modified=false
active_skill_modified=false
automatic_pr_update_performed=false
merge_or_deploy_performed=false
```

## Candidate inputs

```text
output/phase5-continuous-loop/s4-candidate-optimizer-20260613_054510/run-20260613_104130/candidate_descriptions.json
sha256=983ed97fdf22199dc5835cbac3771af0c5aacb25a03b9dcfdfd334f3aeeb0b0e

output/phase5-continuous-loop/s4-candidate-optimizer-20260613_054510/run-20260613_104130/candidate_only_report.sanitized.json
sha256=f2b0de72de787474c4d8ca229ad0b25c5554216e2fa868fadd6b9ffca171cfe9

output/phase5-continuous-loop/s4-candidate-optimizer-20260613_054510/run-20260613_104130/candidate.diff
sha256=07e55738f70f8bff7328df6f106fe4bf27e9245746d8abee6bf64091364e8b75
```

## Future apply gate

Future apply requires separate approval, backup, checksum, read-back, independent review, tests, and rollback plan.

Suggested future phrase:

```text
S4 candidate active apply GO — backup active tool descriptions, apply reviewed candidate only, checksum/read-back/pytest verification; no automatic PR update, no merge/deploy, rollback plan required.
```
