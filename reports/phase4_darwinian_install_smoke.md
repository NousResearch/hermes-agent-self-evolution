# Phase 4 Darwinian Install/CLI Smoke Report

- Created: 2026-06-04 22:14 CEST
- Status: **PASS**
- Mode: isolated dependency smoke
- Scope: Darwinian Evolver 설치 및 실행 가능성 검증

## 결론

Darwinian Evolver는 별도 격리 경로에 clone/install 되었고, CLI help 실행, 핵심 모듈 import smoke, no-network in-process mini evolution loop가 통과했다.

이번 실행은 **Darwinian 설치/실행 smoke**이며, 다음은 수행하지 않았다.

- Hermes Agent source mutation 없음
- HSE native code에서 Darwinian import 없음
- Darwinian을 Hermes source repo에 대해 mutation 실행하지 않음
- Anthropic/OpenAI/Google API backed example problem 실행 없음
- benchmark/API spend 없음
- credentials 출력 없음

## 설치/실행 위치

| 항목 | 값 |
|---|---|
| Upstream repo | `https://github.com/imbue-ai/darwinian_evolver.git` |
| Local repo | `/Users/snw/.hermes/evolution/repos/darwinian_evolver` |
| Upstream HEAD | `7f12365d2059c47e29068a5a6f498a293148d2a9` |
| Isolated venv | `/Users/snw/.hermes/evolution/venvs/darwinian-evolver` |
| Package version | `darwinian-evolver==0.1.0` |
| License first line | `GNU AFFERO GENERAL PUBLIC LICENSE` |
| Smoke run dir | `/Users/snw/.hermes/evolution/darwinian-smoke/20260604T221240` |

## 설치 결과

실행:

```bash
uv pip install --python /Users/snw/.hermes/evolution/venvs/darwinian-evolver/bin/python -e /Users/snw/.hermes/evolution/repos/darwinian_evolver
```

주요 설치 패키지:

| Package | Version |
|---|---:|
| darwinian-evolver | 0.1.0 |
| anthropic | 0.105.2 |
| openai | 2.41.0 |
| google-genai | 2.8.0 |
| numpy | 2.4.6 |
| pydantic | 2.13.4 |
| jinja2 | 3.1.6 |
| func-timeout | 4.3.5 |
| fsspec | 2026.4.0 |

주의: `google-genai`는 upstream package dependency로 **격리 Darwinian venv 안에만 설치**되었다. Google/Gemini API call은 수행하지 않았다.

## Smoke checks

| Check | Result | Evidence |
|---|---|---|
| `git ls-remote` HEAD 확인 | PASS | `7f12365d2059c47e29068a5a6f498a293148d2a9` |
| clone/local repo presence | PASS | `/Users/snw/.hermes/evolution/repos/darwinian_evolver` |
| isolated venv | PASS | Python 3.11.1 |
| editable install | PASS | `darwinian-evolver==0.1.0` |
| CLI help | PASS | `parrot`, `circle_packing`, `multiplication_verifier`, `--num_iterations`, `--output_dir` 확인 |
| CLI help SHA-256 | PASS | `6c2d0f22556b031e9df088c91012ae84a1bf984b97e33960c43f1dbd7d95f68f` |
| import smoke | PASS | 5 modules loaded |
| no-network mini loop | PASS | iteration 0→1, best score 0.0→1.0, population 1→2 |

Import smoke 대상:

- `darwinian_evolver`
- `darwinian_evolver.problem`
- `darwinian_evolver.evolve_problem_loop`
- `darwinian_evolver.git_based_problem`
- `darwinian_evolver.problems.registry`

## No-network mini loop 결과

검증용 mini problem은 Darwinian core loop만 직접 사용했다.

- Initial organism: `value=0`
- Evaluator: `value >= 1`이면 score `1.0`, 아니면 failure case 반환
- Mutator: `value + 1`
- Iterations observed: `[0, 1]`
- Initial best score: `0.0`
- Final best score: `1.0`
- Population size: `1 → 2`
- Mutation calls in iteration 1: `1`

이 loop는 LLM/API/network를 호출하지 않는 deterministic smoke이다.

## Preserved boundaries

- HSE Phase 4 scaffold는 Darwinian을 import하지 않는다.
- HSE Phase 4 scaffold는 Darwinian을 subprocess로 실행하지 않는다.
- Darwinian은 Hermes source repo를 대상으로 mutation하지 않았다.
- API-backed example problems(`parrot`, `multiplication_verifier`, `circle_packing`)는 실행하지 않았다.
- 현재 검증은 availability/basic execution smoke이며, 실제 code evolution approval로 확장하지 않는다.

## Verdict

**PASS.** Darwinian Evolver isolated install, CLI help smoke, module import smoke, and no-network in-process evolution loop passed.

다만 이 결과는 “Darwinian을 사용할 준비가 되었는가?”에 대한 초기 실행 가능성 검증이다. Hermes source mutation, API-backed benchmark, fork/PR publication은 여전히 별도 phase gate로 다루는 것이 안전하다.
