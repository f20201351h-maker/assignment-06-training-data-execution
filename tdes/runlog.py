"""run.log writer shared by the orchestrator and the child processes (append + flush)."""
import datetime
import os
from pathlib import Path


def log(artifacts, msg, echo=True):
    p = Path(artifacts) / "run.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"{ts} [pid {os.getpid()}] {msg}"
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
        f.flush()
    if echo:
        print(line, flush=True)


def check(artifacts, name, ok, detail=""):
    log(artifacts, f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    return ok
