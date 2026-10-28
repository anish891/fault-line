"""Sandboxes: where actions execute and where state is snapshotted/restored.

LocalDirSandbox is a working reference implementation for development and tests.
It runs `bash` actions directly on your machine inside a temp dir, so only use
it with trusted/scripted policies. For real LLM agents use a container sandbox.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol


class Sandbox(Protocol):
    def exec(self, action: dict) -> str: ...
    def read_file(self, path: str) -> str | None: ...
    def snapshot(self, snap_id: str) -> None: ...
    def restore(self, snap_id: str) -> None: ...
    def close(self) -> None: ...


class LocalDirSandbox:
    """Filesystem-only state. Snapshots are directory copies under `snap_root`."""

    def __init__(self, snap_root: Path | str, timeout: float = 10.0):
        self.snap_root = Path(snap_root)
        self.snap_root.mkdir(parents=True, exist_ok=True)
        self.workdir = Path(tempfile.mkdtemp(prefix="faultline-"))
        self.timeout = timeout

    def _safe(self, rel: str) -> Path:
        root = self.workdir.resolve()
        p = (root / rel).resolve()
        if not p.is_relative_to(root):
            raise ValueError(f"path escapes sandbox: {rel}")
        return p

    def exec(self, action: dict) -> str:
        tool, args = action.get("tool"), action.get("args", {})
        try:
            if tool == "write":
                p = self._safe(args["path"])
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(args["content"])
                return f"wrote {len(args['content'])} bytes to {args['path']}"
            if tool == "bash":
                r = subprocess.run(args["cmd"], shell=True, cwd=self.workdir,
                                   capture_output=True, text=True, timeout=self.timeout)
                return (r.stdout + r.stderr).strip()
            if tool == "finish":
                return "done"
            return f"ERROR: unknown tool {tool!r}"
        except Exception as e:  # tool errors are observations, not crashes
            return f"ERROR: {type(e).__name__}: {e}"

    def read_file(self, path: str) -> str | None:
        p = self._safe(path)
        return p.read_text() if p.exists() else None

    def snapshot(self, snap_id: str) -> None:
        dest = self.snap_root / snap_id
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self.workdir, dest)

    def restore(self, snap_id: str) -> None:
        src = self.snap_root / snap_id
        if not src.exists():
            raise FileNotFoundError(f"no snapshot {snap_id!r} in {self.snap_root}")
        for child in self.workdir.iterdir():
            shutil.rmtree(child) if child.is_dir() else child.unlink()
        shutil.copytree(src, self.workdir, dirs_exist_ok=True)

    def close(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)


class DockerSandbox:
    """TODO (week 2). Plan:
      - `docker run -d --network none` a task image; exec actions via `docker exec`.
      - snapshot(): `docker commit <ctr> faultline/<snap_id>` (layers are shared, so cheap).
      - restore(): start a fresh container from that image.
      - GC old snapshot images once an attribution finishes.
    Later optimisation: Firecracker/overlayfs for ms-level fork (see DeltaBox paper).
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DockerSandbox is not implemented yet")
