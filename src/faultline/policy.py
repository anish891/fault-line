"""The policy interface. Anything that can pick the next action fits.

To plug in a real LLM, implement `act` so that it (a) renders `messages` into a
model call with a fixed temperature > 0, and (b) parses the reply into an action
dict. Sampling randomness must come from the model's own sampling; `rng` is
provided for policies (like the toy one) that need seeded randomness.

Actions are plain dicts:
    {"tool": "bash",   "args": {"cmd": "ls"}}
    {"tool": "write",  "args": {"path": "a.txt", "content": "hi"}}
    {"tool": "finish", "args": {}}
"""
from __future__ import annotations

import random
from typing import Protocol


class Policy(Protocol):
    policy_id: str

    def act(self, messages: list[dict], rng: random.Random) -> dict: ...
