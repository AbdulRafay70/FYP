"""Run records for reproducibility: config JSON + epoch-wise CSV log per model."""

import csv
import json
import os
import platform
import subprocess
from datetime import datetime

import torch


def environment_info() -> dict:
    """Hardware / software actually used for the run (for the thesis experimental setup)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=os.path.dirname(__file__)).stdout.strip()
    except Exception:
        commit = ""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
    }


def save_run_config(output_dir: str, name: str, config: dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}_config.json")
    with open(path, "w") as f:
        json.dump({"model": name, "config": config, "environment": environment_info()},
                  f, indent=2, default=str)
    return path


class EpochLogger:
    """Appends one row per epoch to <output_dir>/<name>_log.csv."""

    def __init__(self, output_dir: str, name: str):
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, f"{name}_log.csv")
        self._fields = None
        if os.path.exists(self.path):
            os.remove(self.path)

    def log(self, **row):
        row = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}
        new = self._fields is None
        if new:
            self._fields = list(row.keys())
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fields)
            if new:
                w.writeheader()
            w.writerow(row)
