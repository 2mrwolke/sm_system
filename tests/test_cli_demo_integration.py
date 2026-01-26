import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_cli_demo_runs_and_saves_npz(tmp_path):
    """Integration test: run the CLI demo end-to-end and validate basic invariants."""

    out_path = tmp_path / "demo_run.npz"

    # Keep the demo intentionally small so it runs quickly in CI.
    cmd = [
        sys.executable,
        "-m",
        "sm_system",
        "demo",
        "--out",
        str(out_path),
        "--samplerate",
        "48000",
        "--max-order",
        "3",
        "--probes-n",
        "8",
        "--probes-batches",
        "1",
        "--f-min",
        "200",
        "--f-max",
        "8000",
        "--signal-time",
        "0.003",
        "--transient-time",
        "0.002",
        "--seed",
        "0",
    ]

    # Ensure the subprocess can import the package from the source checkout.
    src = str((Path(__file__).resolve().parents[1] / "src").resolve())
    env = os.environ.copy()
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
    assert out_path.exists(), "CLI demo did not create the expected .npz output"

    # Validate saved contents.
    data = np.load(out_path, allow_pickle=False)
    assert set(data.files) >= {"y", "samplerate", "cfg_json"}

    y = data["y"]
    sr = int(np.asarray(data["samplerate"]).item())
    cfg_json = np.asarray(data["cfg_json"]).item()
    cfg = json.loads(cfg_json)

    # Basic invariants: shape, samplerate consistency, finite output.
    assert sr == 48000
    assert int(cfg["samplerate"]) == 48000

    assert y.ndim == 3  # (batches, signals_per_batch, samples)
    assert y.shape[0] == 1
    assert y.shape[1] > 0
    assert y.shape[2] > 0
    assert np.isfinite(y).all(), "Demo output contains NaN/Inf values"
