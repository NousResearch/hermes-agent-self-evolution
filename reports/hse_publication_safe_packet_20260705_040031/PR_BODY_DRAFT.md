# HSE sanitized publication-safe evidence packet

This PR, if created later, should be treated as a human-review packet only.

## Scope

- Adds a sanitized, path-limited HSE evidence packet under `reports/hse_publication_safe_packet_20260705_040031/`.
- Preserves no-auto-merge and human-review boundary.
- Does not claim upstream merge completion or CI pass until checks are attached and verified.

## Current local evidence summary

- strict status: `HSE_PROJECT_STRICT_COMPLETE`
- manifest status: `HSE_PROJECT_STRICT_COMPLETE`
- highest strict complete phase: `5`
- local PLAN SHA-256: `edfdf202dd164c9d4618909f18ecff1016dd187d98e3bd46ac784613a6e6b662`
- remote PLAN SHA-256: `5928030b470710480a4c0fd7e702af1b2352c9470adddc437c3bc965a23a71f8`
- local/remote PLAN equal: `False`

## Required reviewer checks before merge

- Confirm all PR files are under `reports/hse_publication_safe_packet_20260705_040031/`.
- Confirm no raw local home paths or credentials are present.
- Confirm CI/check state; missing checks are not pass.
- Confirm `autoMergeRequest == null`.
- Confirm human review accepts the local-operational PLAN caveat.

## Explicitly not performed by packet generation

- GitHub write or PR creation
- branch push
- merge or auto-merge
- deploy/publication
- provider/model spend
- gateway restart/reload
- cron mutation
