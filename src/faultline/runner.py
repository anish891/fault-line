"""Run an agent loop, record it, and fork it from any recorded step."""
from __future__ import annotations

import json
import random
from typing import Callable

from .db import Store
from .models import Intervention, Task
from .policy import Policy
from .sandbox import Sandbox


def steps_to_messages(task: Task, steps: list[dict]) -> list[dict]:
    """Rebuild the chat history the agent saw, from recorded steps."""
    msgs = [{"role": "user", "content": task.prompt}]
    for s in steps:
        msgs.append({"role": "assistant", "content": json.dumps(s["action"], sort_keys=True)})
        msgs.append({"role": "tool", "content": s["observation"]})
    return msgs


def _loop(store: Store, run_id: str, task: Task, policy: Policy, sb: Sandbox,
          messages: list[dict], start_idx: int, seed: int, max_steps: int,
          intervention: Intervention | None, save_snapshots: bool) -> bool:
    rng = random.Random(seed)
    idx, finished = start_idx, False
    while idx < max_steps:
        snap_id = None
        if save_snapshots:
            snap_id = f"{run_id}-{idx}"
            sb.snapshot(snap_id)  # state BEFORE step idx: lets us re-decide step idx later

        if intervention and intervention.kind == "force_action" and idx == intervention.step:
            action = intervention.action
        else:
            action = policy.act(messages, rng)

        obs = sb.exec(action)
        if intervention and intervention.kind == "replace_observation" and idx == intervention.step:
            obs = intervention.observation

        store.add_step(run_id, idx, snap_id, action, obs)
        messages.append({"role": "assistant", "content": json.dumps(action, sort_keys=True)})
        messages.append({"role": "tool", "content": obs})
        idx += 1
        if action.get("tool") == "finish":
            finished = True
            break

    success = bool(task.check(sb))
    store.finish_run(run_id, "finished" if finished else "max_steps", success)
    return success


def run_fresh(store: Store, task: Task, policy: Policy, make_sandbox: Callable[[], Sandbox],
              *, seed: int, max_steps: int = 20, save_snapshots: bool = True) -> str:
    """Run the agent from scratch, recording every step (+ a snapshot before each)."""
    run_id = store.new_run(task_id=task.task_id, policy_id=policy.policy_id, seed=seed)
    sb = make_sandbox()
    try:
        task.setup(sb)
        _loop(store, run_id, task, policy, sb, [{"role": "user", "content": task.prompt}],
              0, seed, max_steps, None, save_snapshots)
    finally:
        sb.close()
    return run_id


def run_fork(store: Store, task: Task, policy: Policy, make_sandbox: Callable[[], Sandbox],
             parent_run_id: str, k: int, *, seed: int, intervention: Intervention | None = None,
             max_steps: int = 20, save_snapshots: bool = False) -> str:
    """Fork `parent_run_id` at step k: keep steps [0,k), restore the world as it was
    before step k, then let the policy continue live (resampling step k onward)."""
    steps = store.full_steps(parent_run_id)
    if not 0 <= k < len(steps):
        raise ValueError(f"fork step {k} out of range for run with {len(steps)} steps")
    snap = steps[k]["snap_before"]
    if snap is None:
        raise ValueError(f"step {k} has no snapshot; record the parent with save_snapshots=True")

    run_id = store.new_run(
        task_id=task.task_id, policy_id=policy.policy_id, seed=seed,
        parent_run_id=parent_run_id, fork_step=k,
        intervention=intervention.to_json() if intervention else None)
    sb = make_sandbox()
    try:
        sb.restore(snap)
        _loop(store, run_id, task, policy, sb, steps_to_messages(task, steps[:k]),
              k, seed, max_steps, intervention, save_snapshots)
    finally:
        sb.close()
    return run_id
