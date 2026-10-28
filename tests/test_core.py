from functools import partial

import pytest

from faultline import Intervention, LocalDirSandbox, Store, attribute, run_fork, run_fresh
from faultline.stats import wilson_interval
from faultline.toy import CULPRIT_STEP, ToyPolicy, toy_task


@pytest.fixture
def env(tmp_path):
    store = Store()
    make = partial(LocalDirSandbox, tmp_path / "snaps")
    return store, make, toy_task(), ToyPolicy(0.7)


def first_failed_run(env):
    store, make, task, policy = env
    for seed in range(200):
        rid = run_fresh(store, task, policy, make, seed=seed)
        if not store.get_run(rid)["success"]:
            return rid
    raise AssertionError("no failing seed found")


def test_wilson_bounds():
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0.1 < hi < 0.2
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_snapshot_restore_roundtrip(tmp_path):
    sb = LocalDirSandbox(tmp_path / "s")
    sb.exec({"tool": "write", "args": {"path": "a.txt", "content": "one"}})
    sb.snapshot("s1")
    sb.exec({"tool": "write", "args": {"path": "a.txt", "content": "two"}})
    sb.exec({"tool": "write", "args": {"path": "b.txt", "content": "x"}})
    sb.restore("s1")
    assert sb.read_file("a.txt") == "one"
    assert sb.read_file("b.txt") is None
    sb.close()


def test_sandbox_blocks_path_escape(tmp_path):
    sb = LocalDirSandbox(tmp_path / "s")
    out = sb.exec({"tool": "write", "args": {"path": "../evil.txt", "content": "x"}})
    assert out.startswith("ERROR")
    sb.close()


def test_fork_keeps_prefix_and_world_state(env):
    store, make, task, policy = env
    parent = first_failed_run(env)
    child = run_fork(store, task, policy, make, parent, 3, seed=1)
    steps = store.full_steps(child)
    assert [s["idx"] for s in steps] == [0, 1, 2, 3, 4]
    # prefix is inherited unchanged, including the bad write at step 2
    assert steps[2]["action"] == store.full_steps(parent)[2]["action"]
    # world was restored: step 3 `cat answer.txt` sees the parent's bad answer
    assert steps[3]["observation"] == "13"
    assert store.get_run(child)["success"] == 0


def test_force_action_repairs_failure(env):
    store, make, task, policy = env
    parent = first_failed_run(env)
    fix = {"tool": "write", "args": {"path": "answer.txt", "content": "12"}}
    child = run_fork(store, task, policy, make, parent, CULPRIT_STEP, seed=1,
                     intervention=Intervention("force_action", CULPRIT_STEP, action=fix))
    assert store.get_run(child)["success"] == 1


def test_replace_observation_is_shown_to_agent(env):
    store, make, task, policy = env
    parent = first_failed_run(env)
    child = run_fork(store, task, policy, make, parent, 1, seed=1,
                     intervention=Intervention("replace_observation", 1, observation="9\n9\n"))
    assert store.full_steps(child)[1]["observation"] == "9\n9\n"


@pytest.mark.parametrize("search", ["linear", "bisect"])
def test_attribution_finds_planted_culprit(env, search):
    store, make, task, policy = env
    run_id = first_failed_run(env)
    res = attribute(store, task, policy, make, run_id, n=20, search=search)
    assert res.culprit_step == CULPRIT_STEP, res.reason
    assert res.baseline.p > 0.5          # control arm is healthy
    assert res.probes[CULPRIT_STEP + 1].successes == 0   # sealed after the culprit


def test_bisect_uses_fewer_rollouts_than_linear(env):
    store, make, task, policy = env
    run_id = first_failed_run(env)
    lin = attribute(store, task, policy, make, run_id, n=10, search="linear")
    bis = attribute(store, task, policy, make, run_id, n=10, search="bisect")
    assert lin.culprit_step == bis.culprit_step == CULPRIT_STEP
    assert bis.n_rollouts <= lin.n_rollouts


def test_rejects_successful_run(env):
    store, make, task, policy = env
    for seed in range(200):
        rid = run_fresh(store, task, policy, make, seed=seed)
        if store.get_run(rid)["success"]:
            break
    with pytest.raises(ValueError):
        attribute(store, task, policy, make, rid)


def test_hard_task_is_inconclusive(tmp_path):
    store = Store()
    make = partial(LocalDirSandbox, tmp_path / "snaps")
    task, policy = toy_task(), ToyPolicy(0.05)   # almost always fails: no baseline to compare
    rid = run_fresh(store, task, policy, make, seed=0)
    assert not store.get_run(rid)["success"]
    res = attribute(store, task, policy, make, rid, n=20)
    assert res.culprit_step is None and "too hard" in res.reason
