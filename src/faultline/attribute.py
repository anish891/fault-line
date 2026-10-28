"""Counterfactual attribution by resampling.

Definitions. For a failed run with steps 0..T-1, let
    q(k) = P(success | steps [0,k) held fixed, steps k.. resampled from the policy).
q(0) is a full re-run: the baseline success rate p0 (the control arm).

If one step c "sealed" the failure, then q(k) stays near p0 while k <= c (step c
is still being resampled) and collapses once k > c (the bad step is now fixed in
the prefix). So the commitment point is m = min{k : q(k) is significantly below p0}
and the culprit is step m-1. This is the "point of commitment" idea: the step that
did the damage is the LAST one whose resampling can still rescue the run.

Search: q(k) is assumed roughly monotone non-increasing in k for a failed run, so
we can binary-search for m (like `git bisect`) instead of probing every step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .db import Store
from .models import Task
from .policy import Policy
from .runner import run_fork
from .sandbox import Sandbox
from .stats import wilson_interval


@dataclass
class ProbeResult:
    k: int
    n: int
    successes: int

    @property
    def p(self) -> float:
        return self.successes / self.n if self.n else 0.0

    @property
    def lo(self) -> float:
        return wilson_interval(self.successes, self.n)[0]

    @property
    def hi(self) -> float:
        return wilson_interval(self.successes, self.n)[1]


@dataclass
class Attribution:
    run_id: str
    culprit_step: int | None
    reason: str
    baseline: ProbeResult | None = None
    probes: dict[int, ProbeResult] = field(default_factory=dict)
    n_rollouts: int = 0

    def curve(self) -> list[ProbeResult]:
        pts = ([self.baseline] if self.baseline else []) + [self.probes[k] for k in sorted(self.probes) if k != 0]
        return pts


def probe(store: Store, task: Task, policy: Policy, make_sandbox: Callable[[], Sandbox],
          run_id: str, k: int, n: int, *, seed0: int = 1000, max_steps: int = 20) -> ProbeResult:
    """Resample the run from step k, n times, and count successes."""
    wins = 0
    for i in range(n):
        child = run_fork(store, task, policy, make_sandbox, run_id, k,
                         seed=seed0 * 1_000_003 + k * 1009 + i, max_steps=max_steps)
        wins += store.get_run(child)["success"]
    store.record_probe(run_id, k, "resample", n, wins)
    return ProbeResult(k, n, wins)


def _dropped(q: ProbeResult, base: ProbeResult, delta: float) -> bool:
    """Significant drop: point estimate is >= delta below baseline AND the upper
    Wilson bound of q is still below the baseline point estimate."""
    return q.p <= base.p - delta and q.hi < base.p


def attribute(store: Store, task: Task, policy: Policy, make_sandbox: Callable[[], Sandbox],
              run_id: str, *, n: int = 20, delta: float = 0.3, search: str = "bisect",
              seed0: int = 1000, max_steps: int = 20) -> Attribution:
    run = store.get_run(run_id)
    if run["success"]:
        raise ValueError("attribute() expects a FAILED run")
    T = len(store.full_steps(run_id))
    result = Attribution(run_id=run_id, culprit_step=None, reason="")

    def get(k: int) -> ProbeResult:
        if k not in result.probes:
            result.probes[k] = probe(store, task, policy, make_sandbox, run_id, k, n,
                                     seed0=seed0, max_steps=max_steps)
            result.n_rollouts += n
        return result.probes[k]

    base = get(0)
    result.baseline = base
    if base.p < delta:
        result.reason = (f"baseline success {base.p:.0%} is below delta={delta:.0%}; "
                         "the task is too hard to attribute reliably")
        return result
    if T < 2:
        result.reason = "run has fewer than 2 steps"
        return result

    m: int | None = None
    if search == "linear":
        for k in range(1, T):
            if _dropped(get(k), base, delta):
                m = k
                break
    elif search == "bisect":
        if _dropped(get(T - 1), base, delta):
            lo, hi = 1, T - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if _dropped(get(mid), base, delta):
                    hi = mid
                else:
                    lo = mid + 1
            m = lo
    else:
        raise ValueError("search must be 'linear' or 'bisect'")

    if m is None:
        result.reason = "no significant drop found: failure not attributable to a single step"
    else:
        result.culprit_step = m - 1
        result.reason = (f"success fell from {base.p:.0%} (baseline) to {get(m).p:.0%} "
                         f"once steps [0,{m}) were fixed")
    return result
