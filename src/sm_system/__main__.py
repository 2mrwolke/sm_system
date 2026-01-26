"""Command-line entry point for sm_system.

This module provides a small, dependency-light reproduction script so the
repository can be exercised without relying on notebooks.

Examples
--------
Run a small end-to-end demo (probe generation -> simulation -> save results):

    python -m sm_system demo --out demo_run.npz

Or, if installed:

    sm-system demo --out demo_run.npz
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .sim import Simulation
from .sm import SM_System


def _cfg_to_json_dict(cfg: Any) -> Dict[str, Any]:
    """Convert Configuration (slots dataclass) to a JSON-serializable dict."""
    # Configuration is a slots dataclass; asdict() works but includes private keys.
    # Prefer the explicit key list if present.
    if hasattr(cfg, "_keys"):
        raw = {k: getattr(cfg, k) for k in getattr(cfg, "_keys")}
    else:
        raw = asdict(cfg)  # pragma: no cover

    def _coerce(v: Any) -> Any:
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.ndarray,)):
            return v.tolist()
        return v

    return {k: _coerce(v) for k, v in raw.items()}


def _build_demo_sm(sim: Simulation) -> SM_System:
    """Build a small SM model for demo purposes.

    The demo uses continuous-time analog prototypes wrapped as callables.
    This is meant for a quick reproduction run, not for high-throughput.
    """

    sr = sim.cfg.samplerate

    # Helper to obtain a callable LTI block.
    def lti_callable(name: str, **kwargs):
        # The LTI decorator supports is_callable + sr.
        return sim.lti.system[name](is_callable=True, sr=sr, **kwargs)

    ident = lambda x: x

    # 1st-order branch: unity (linear pass-through)
    s1 = SM_System(pre=ident, nl=ident, post=ident, name="s1")

    # 2nd-order branch: mild nonlinearity with pre/post shaping
    lp2 = lti_callable("Low-pass 2nd", wc=2 * np.pi * 1800.0, q=0.707)
    ap1 = lti_callable("All-pass 1st", wc=2 * np.pi * 1200.0)
    s2 = SM_System(pre=lp2, nl=lambda x: x**2, post=ap1, name="s2")

    # 3rd-order branch: stronger shaping to make spectral products visible
    hp2 = lti_callable("High-pass 2nd", wc=2 * np.pi * 600.0, q=0.707)
    lp1 = lti_callable("Low-pass 1st", wc=2 * np.pi * 4200.0)
    s3 = SM_System(pre=hp2, nl=lambda x: x**3, post=lp1, name="s3")

    return s1 + s2 + s3


def _cmd_demo(args: argparse.Namespace) -> int:
    sim = Simulation(rm_dc=True, rm_transients=True)

    # Keep the demo run small by default, but configurable.
    sim.cfg.update("samplerate", args.samplerate)
    sim.cfg.update("max_order", args.max_order)
    sim.cfg.update("probes_n", args.probes_n)
    sim.cfg.update("probes_batches", args.probes_batches)
    sim.cfg.update("probes_fbase", args.f_min)
    sim.cfg.update("probes_fvoice", args.f_max)
    sim.cfg.update("probes_signaltime", args.signal_time)
    sim.cfg.update("probes_transient_time", args.transient_time)
    sim.cfg.update("seed", args.seed)
    sim.build_harmonic_probes()

    sm = _build_demo_sm(sim)
    sim.set_sm(sm)

    y = sim.run()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg_json = json.dumps(_cfg_to_json_dict(sim.cfg), indent=2, sort_keys=True)
    np.savez_compressed(
        out_path,
        y=y,
        samplerate=np.array(sim.cfg.samplerate, dtype=int),
        cfg_json=np.array(cfg_json),
    )

    print("sm_system demo run complete")
    print(f"  y shape: {tuple(y.shape)} (batches, signals_per_batch, samples)")
    print(f"  samplerate: {sim.cfg.samplerate} Hz")
    print(f"  saved: {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sm-system", description="sm_system CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser(
        "demo",
        help="Run a small end-to-end demo (probe generation -> SM simulation -> save).",
    )
    demo.add_argument("--out", default="demo_run.npz", help="Output .npz path")
    demo.add_argument("--samplerate", type=int, default=96_000)
    demo.add_argument("--max-order", type=int, default=3, dest="max_order")
    demo.add_argument("--probes-n", type=int, default=16, dest="probes_n")
    demo.add_argument("--probes-batches", type=int, default=2, dest="probes_batches")
    demo.add_argument("--f-min", type=int, default=200, dest="f_min")
    demo.add_argument("--f-max", type=int, default=16_000, dest="f_max")
    demo.add_argument(
        "--signal-time",
        type=float,
        default=0.01,
        dest="signal_time",
        help="Steady-state time in seconds (will be quantized to an FFT-friendly length).",
    )
    demo.add_argument(
        "--transient-time",
        type=float,
        default=0.01,
        dest="transient_time",
        help="Transient time in seconds (removed if rm_transients is enabled).",
    )
    demo.add_argument("--seed", type=int, default=42)
    demo.set_defaults(func=_cmd_demo)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
