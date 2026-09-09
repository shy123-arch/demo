#!/usr/bin/env python3
"""Print only the current MolmoSpaces teleop task description.

This is intended for a dedicated terminal while collecting PICO demos. It
follows the newest main log and prints only real task lines, e.g.

    Pick up the bottle

No health/stat/control lines are emitted.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TASK_RE = re.compile(r"\[TELEOP_TASK\]\s*(.+?)\s*$")
SKIP_TASK_PREFIXES = (
    "Open PICO ",
    "Hold RIGHT ",
    "LEFT X ",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/mnt/f/molospace/run_outputs"),
        help="Directory containing molmospaces_main_pico_*.log files.",
    )
    parser.add_argument("--pattern", default="molmospaces_main_pico_*.log")
    parser.add_argument("--poll-s", type=float, default=0.5)
    parser.add_argument(
        "--no-initial",
        action="store_true",
        help="Do not print the latest existing task when the watcher starts.",
    )
    return parser.parse_args()


def clean_line(line: str) -> str:
    return ANSI_RE.sub("", line).replace("\r", "").strip()


def task_from_line(line: str) -> str | None:
    match = TASK_RE.search(clean_line(line))
    if not match:
        return None
    task = match.group(1).strip()
    if not task or any(task.startswith(prefix) for prefix in SKIP_TASK_PREFIXES):
        return None
    return task


def newest_log(log_dir: Path, pattern: str) -> Path | None:
    logs = [path for path in log_dir.glob(pattern) if path.is_file()]
    if not logs:
        return None
    return max(logs, key=lambda p: (p.stat().st_mtime, p.name))


def latest_task(path: Path) -> str | None:
    last: str | None = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                task = task_from_line(line)
                if task:
                    last = task
    except OSError:
        return None
    return last


def main() -> None:
    args = parse_args()
    current_path: Path | None = None
    current_pos = 0

    while True:
        path = newest_log(args.log_dir, args.pattern)
        if path is None:
            time.sleep(args.poll_s)
            continue

        if path != current_path:
            current_path = path
            current_pos = path.stat().st_size
            if not args.no_initial:
                task = latest_task(path)
                if task:
                    print(task, flush=True)

        try:
            size = path.stat().st_size
            if size < current_pos:
                current_pos = 0
            with path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(current_pos)
                for line in f:
                    task = task_from_line(line)
                    if task:
                        print(task, flush=True)
                current_pos = f.tell()
        except OSError:
            current_path = None
            current_pos = 0

        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
