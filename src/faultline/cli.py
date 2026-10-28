from __future__ import annotations

import argparse
import tempfile
from functools import partial

from .attribute import attribute
from .db import Store
from .runner import run_fresh
from .sandbox import LocalDirSandbox
from .toy import CULPRIT_STEP, ToyPolicy, toy_task


def _bar(p: float, width: int = 30) -> str:
    return "#" * round(p * width) + "." * (width - round(p * width))


def demo(args: argparse.Namespace) -> None:
    store = Store(args.db)
    task, policy = toy_task(), ToyPolicy(args.p_correct)
    with tempfile.TemporaryDirectory(prefix="faultline-snaps-") as snaps:
        make = partial(LocalDirSandbox, snaps)
        run_id = None
        for seed in range(500):
            rid = run_fresh(store, task, policy, make, seed=seed)
            if not store.get_run(rid)["success"]:
                run_id = rid
                break
        if run_id is None:
            raise SystemExit("could not find a failing run; lower --p-correct")
        print(f"failed run {run_id} (seed {seed}); planted culprit = step {CULPRIT_STEP}\n")

        res = attribute(store, task, policy, make, run_id, n=args.n, search=args.search)
        print(f"P(success | steps [0,k) fixed, rest resampled)   [search={args.search}]")
        for q in res.curve():
            tag = "  <- baseline (control)" if q.k == 0 else ""
            print(f"  k={q.k}  {_bar(q.p)} {q.p:4.0%}  [{q.lo:.2f}, {q.hi:.2f}]{tag}")
        print()
        if res.culprit_step is None:
            print(f"verdict: inconclusive ({res.reason})")
        else:
            print(f"verdict: step {res.culprit_step} caused the failure ({res.reason})")
        print(f"cost: {res.n_rollouts} rollouts")


def runs(args: argparse.Namespace) -> None:
    for r in Store(args.db).list_runs():
        parent = f" fork of {r['parent_run_id']}@{r['fork_step']}" if r["parent_run_id"] else ""
        print(f"{r['run_id']}  {r['status']:<10} success={r['success']}{parent}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="faultline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="attribute a planted failure in the toy world")
    d.add_argument("--db", default=":memory:")
    d.add_argument("--n", type=int, default=20, help="resamples per probe")
    d.add_argument("--p-correct", type=float, default=0.7)
    d.add_argument("--search", choices=["bisect", "linear"], default="linear")
    d.set_defaults(fn=demo)

    r = sub.add_parser("runs", help="list recorded runs")
    r.add_argument("--db", required=True)
    r.set_defaults(fn=runs)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
