# Fault Line

Counterfactual failure attribution for agent runs: **find the step that actually
caused the failure, and prove it with reruns.**

```
P(success | steps [0,k) fixed, rest resampled)
  k=0  #####################.........  70%  <- baseline (control)
  k=1  ######################........  75%
  k=2  ######################........  75%
  k=3  ..............................   0%     <- failure is sealed here
verdict: step 2 caused the failure
```

## How it works

1. **Record** a run: every action, observation, and a world snapshot *before* each step.
2. **Fork** at step `k`: restore the snapshot, keep the history `[0,k)`, let the policy continue live.
3. **Probe**: fork `n` times from step `k` and measure `q(k) = P(success)`.
4. **Control arm**: `q(0)` is a full rerun, i.e. the task's baseline success rate. Without it
   you blame steps for luck.
5. **Attribute**: if one step sealed the failure, `q(k)` stays near baseline while step `c` is
   still being resampled (`k <= c`) and collapses once it is fixed in the prefix (`k > c`).
   The culprit is the step just before the first significant drop. Binary search finds it in
   O(log T) probes, assuming `q(k)` is roughly monotone.

Prior work this builds on (read these, cite them): Causal Agent Replay (CAR), agent-bisect,
culprit-agent-rca, AgenTracer, and the Who&When benchmark. This repo's goal is scale, real
coding tasks, environment snapshots, and comparison against LLM-judge attribution.

## Quickstart

```bash
pip install -e ".[dev]"
pytest                          # 11 tests, no network, no API keys
faultline demo --search linear  # attribute a planted failure in the toy world
```

## Layout

| file | role |
|---|---|
| `db.py` | SQLite store; runs form a tree (`parent_run_id`, `fork_step`) |
| `sandbox.py` | `Sandbox` protocol; `LocalDirSandbox` works, `DockerSandbox` is a TODO |
| `runner.py` | `run_fresh`, `run_fork`, interventions (`force_action`, `replace_observation`) |
| `attribute.py` | probes, Wilson-interval drop test, linear/bisect search |
| `policy.py` | the `Policy` protocol; implement `act()` to plug in an LLM |
| `toy.py` | toy task with a planted ground-truth culprit (validation harness) |

## Roadmap

- [ ] `LLMPolicy` (temperature > 0), wired to a bash-only agent loop (e.g. mini-SWE-agent)
- [ ] `DockerSandbox` with `docker commit` snapshots; snapshot GC
- [ ] Parallel probes (thread pool); cost accounting in tokens and dollars
- [ ] Baselines: LLM-judge attribution, last-error heuristic
- [ ] Benchmark runner: SWE-bench Verified subset, flaky-test filtering, baseline-rate filtering
- [ ] Planted-fault suite beyond the toy (corrupted tool results at known steps)
- [ ] Joint-cause detection (Shapley over step pairs) for failures needing two bad steps
- [ ] Web UI: success-probability curve, fork tree, diff

## Known limitations

- `LocalDirSandbox` runs `bash` on your machine. Use it only with scripted/trusted policies.
- Resampling only changes the *agent's* randomness; nondeterministic tools or tests add noise.
- Single-step attribution can't explain failures that need two bad steps together; it reports
  "inconclusive" rather than guessing.
- The significance rule (`delta`, Wilson bound) is a heuristic. Validate it on planted faults
  before trusting it on real agents.

## Author

Built by [Anish Tejwani](https://anishtejwani.dev).
