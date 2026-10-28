"""A toy world with a PLANTED failure point, for validating the method.

Task: sum the numbers in data.txt and write the result to answer.txt.
The scripted policy takes 5 steps. At step 2 (the decision step) it writes the
right answer with probability p_correct, otherwise a wrong one. Nothing later
repairs a wrong answer, so step 2 is the ground-truth culprit of every failure.
"""
from __future__ import annotations

import json
import random

from .models import Task
from .sandbox import Sandbox

CULPRIT_STEP = 2


def _setup(sb: Sandbox) -> None:
    sb.exec({"tool": "write", "args": {"path": "data.txt", "content": "3\n4\n5\n"}})


def _check(sb: Sandbox) -> bool:
    out = sb.read_file("answer.txt")
    return out is not None and out.strip() == "12"


def toy_task() -> Task:
    return Task(
        task_id="toy-sum",
        prompt="Read data.txt, sum the numbers, write the sum to answer.txt, then finish.",
        setup=_setup,
        check=_check,
    )


class ToyPolicy:
    def __init__(self, p_correct: float = 0.7):
        self.p_correct = p_correct
        self.policy_id = f"toy(p={p_correct})"

    def act(self, messages: list[dict], rng: random.Random) -> dict:
        step = sum(1 for m in messages if m["role"] == "assistant")
        if step == 0:
            return {"tool": "bash", "args": {"cmd": "ls"}}
        if step == 1:
            return {"tool": "bash", "args": {"cmd": "cat data.txt"}}
        if step == CULPRIT_STEP:
            ok = rng.random() < self.p_correct
            return {"tool": "write", "args": {"path": "answer.txt", "content": "12" if ok else "13"}}
        if step == 3:
            return {"tool": "bash", "args": {"cmd": "cat answer.txt"}}
        return {"tool": "finish", "args": {}}
