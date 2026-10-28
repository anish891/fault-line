"""SQLite event store. Runs form a tree: a fork points at its parent run + step."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id        TEXT PRIMARY KEY,
  parent_run_id TEXT REFERENCES runs(run_id),
  fork_step     INTEGER,
  intervention  TEXT,
  task_id       TEXT NOT NULL,
  policy_id     TEXT NOT NULL,
  seed          INTEGER NOT NULL,
  status        TEXT NOT NULL DEFAULT 'running',
  success       INTEGER,
  created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS steps (
  run_id      TEXT NOT NULL REFERENCES runs(run_id),
  idx         INTEGER NOT NULL,
  snap_before TEXT,
  action      TEXT NOT NULL,
  observation TEXT NOT NULL,
  PRIMARY KEY (run_id, idx)
);
CREATE TABLE IF NOT EXISTS probes (
  root_run_id TEXT NOT NULL REFERENCES runs(run_id),
  k           INTEGER NOT NULL,
  kind        TEXT NOT NULL,
  n           INTEGER NOT NULL,
  successes   INTEGER NOT NULL,
  PRIMARY KEY (root_run_id, k, kind)
);
CREATE INDEX IF NOT EXISTS runs_parent ON runs(parent_run_id);
"""


class Store:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def new_run(self, *, task_id: str, policy_id: str, seed: int,
                parent_run_id: str | None = None, fork_step: int | None = None,
                intervention: str | None = None) -> str:
        run_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO runs (run_id, parent_run_id, fork_step, intervention, task_id,"
            " policy_id, seed, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (run_id, parent_run_id, fork_step, intervention, task_id, policy_id, seed, time.time()),
        )
        self.conn.commit()
        return run_id

    def add_step(self, run_id: str, idx: int, snap_before: str | None,
                 action: dict, observation: str) -> None:
        self.conn.execute(
            "INSERT INTO steps (run_id, idx, snap_before, action, observation) VALUES (?,?,?,?,?)",
            (run_id, idx, snap_before, json.dumps(action, sort_keys=True), observation),
        )
        self.conn.commit()

    def finish_run(self, run_id: str, status: str, success: bool) -> None:
        self.conn.execute("UPDATE runs SET status=?, success=? WHERE run_id=?",
                          (status, int(success), run_id))
        self.conn.commit()

    def get_run(self, run_id: str) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"no such run: {run_id}")
        return row

    def full_steps(self, run_id: str) -> list[dict[str, Any]]:
        """Steps of a run, including the inherited prefix from its ancestors."""
        run = self.get_run(run_id)
        steps: list[dict[str, Any]] = []
        if run["parent_run_id"] is not None:
            steps = self.full_steps(run["parent_run_id"])[: run["fork_step"]]
        rows = self.conn.execute(
            "SELECT idx, snap_before, action, observation FROM steps WHERE run_id=? ORDER BY idx",
            (run_id,),
        ).fetchall()
        steps += [{"idx": r["idx"], "snap_before": r["snap_before"],
                   "action": json.loads(r["action"]), "observation": r["observation"]}
                  for r in rows]
        return steps

    def record_probe(self, root_run_id: str, k: int, kind: str, n: int, successes: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO probes (root_run_id, k, kind, n, successes) VALUES (?,?,?,?,?)",
            (root_run_id, k, kind, n, successes),
        )
        self.conn.commit()

    def list_runs(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()

    def close(self) -> None:
        self.conn.close()
