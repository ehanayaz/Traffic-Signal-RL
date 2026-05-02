"""Launch training scripts as subprocesses; tail logs; resolve CSV paths by phase."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Phase = Literal["A", "B", "C"]

REPO = Path(__file__).resolve().parent.parent

TRAIN_SCRIPT: dict[Phase, str] = {
    "A": "train.py",
    "B": "train_phase_b.py",
    "C": "train_phase_c.py",
}


def train_script_path(phase: Phase) -> Path:
    return REPO / TRAIN_SCRIPT[phase]


def csv_path_for_phase(phase: Phase) -> Path:
    if phase == "A":
        return REPO / "runs" / "phase_a_train.csv"
    if phase == "B":
        return REPO / "runs" / "phase_b_train.csv"
    return REPO / "runs" / "phase_c_train.csv"


@dataclass
class TrainJob:
    phase: Phase
    process: subprocess.Popen[str] | None = None
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=500))

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> None:
        self.stop()
        script = REPO / TRAIN_SCRIPT[self.phase]
        env = os.environ.copy()
        cwd = str(REPO)
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(script)],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def pump() -> None:
            assert self.process and self.process.stdout
            for line in self.process.stdout:
                self.log_lines.append(line.rstrip())

        threading.Thread(target=pump, daemon=True).start()

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                self.process.send_signal(signal.SIGINT)
            except Exception:
                self.process.terminate()
        self.process = None
