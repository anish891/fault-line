from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .sandbox import Sandbox


@dataclass(frozen=True)
class Intervention:
    """A single do(.) operation applied at one step of a forked run.

    kind="force_action":        replace the action the policy would take at `step`.
    kind="replace_observation": run the real action, but show the agent `observation`.
    Plain resampling (the default fork) needs no Intervention object.
    """

    kind: str
    step: int
    action: dict | None = None
    observation: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {"kind": self.kind, "step": self.step,
             "action": self.action, "observation": self.observation},
            sort_keys=True,
        )


@dataclass
class Task:
    """A task the agent attempts. `check` is the ground-truth success test."""

    task_id: str
    prompt: str
    setup: Callable[["Sandbox"], None]
    check: Callable[["Sandbox"], bool]
