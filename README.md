# sm_system

Virtual simulation & identification for nonlinear systems with **SM structure**.

**What is an SM system?**  
A system whose output is the sum of parallel polynomial branches. Each branch of order *m* is a cascade of:  
**linear pre‑filter** \(H_{m1}\) → **static nonlinearity** \((\cdot)^m\) → **linear post‑filter** \(H_{m2}\).  
This structure was introduced by Baumgartner & Rugh **[BR75]**.

![SM-System](https://github.com/2mrwolke/sm_system/blob/main/SM.png)

---

## Why this project
**sm_system** provides a compact environment to:
- **Build** SM-structured models by composing linear blocks and static powers.
- **Probe** a DUT or a simulated SM system with carefully chosen **harmonic frequency pairs** that avoid intermodulation conflicts up to a selected order.
- **Simulate** time‑domain responses of the SM structure (or your DUT).
- **Identify** the linear parts \(H_{m1}, H_{m2}\) from measured steady‑state frequency responses using **least squares** with optional constraints.
- **Visualize** results with Bode plots and, if desired, **design FIR filters** that match identified magnitude/phase.

Use it to prototype identification procedures, design measurement campaigns, or sanity‑check whether a DUT is well approximated by an SM model.

---

## Key ideas at a glance
- **SM structure**: output is a sum of branches \(y(t)=\sum_m H_{m2}\{(\,H_{m1}\{x(t)\}\,)^m\}\).
- **Harmonic probes**: dual‑tone stimuli (in batches) whose frequency pairs are chosen so that 2nd/3rd‑order products land in **clean FFT bins** (no collisions up to the order you target).
- **Identification**: from the steady‑state spectra at those bins, you build a linear system \(A\theta=b\) and solve for unknown magnitude/phase responses per branch, optionally applying amplitude/phase constraints.
- **Configuration**: one **dataclass** holds probe settings, FFT sizes, noise/SNR, branch filters, and constraint toggles — making runs reproducible.

---

## Quick start

### Install
This project is packaged as a standard Python distribution (PEP 517/518). Install it with pip:

```bash
#!/usr/bin/env bash
set -euo pipefail

repo_url="https://github.com/2mrwolke/sm_system.git"
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

git clone "$repo_url" "$workdir/sm_system"
cd "$workdir/sm_system"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -e ".[dev,plot,progress]"

pytest -q

python -m sm_system demo --out demo_run.npz
test -s demo_run.npz
ls -lh demo_run.npz
```

Typical runtime dependencies include `numpy` and `scipy`. Optional features use `matplotlib` (plotting) and
`tqdm` (progress bars).

### Minimal example (simulation + identification skeleton)

For an end‑to‑end walk‑through, see **`example.ipynb`** (explains probe generation, indexing of intermod bins, and the LS setup).

### Repro script (no notebook required)

After installing the package (editable or normal), you can run a small end‑to‑end demo from the command line:

```bash
# Works without an entrypoint install
python -m sm_system demo --out demo_run.npz

# Or, after installation, via the console script
sm-system demo --out demo_run.npz
```

The demo writes a compressed **`.npz`** containing the simulated probe responses `y` and a JSON snapshot of the active configuration.

---

## Modules (what lives where)
- `src/sm_system/sm.py` — **`SM_System`**: compose per‑order branches (`add`, `merge`, callable system).
- `src/sm_system/sim.py` — **`Simulation`**: orchestrates configuration, probe generation, runs the system/DUT.
- `src/sm_system/probe.py` — **HarmonicProbeIterator**: creates frequency‑pair batches that avoid collisions up to a chosen order.
- `src/sm_system/lti.py` — **LTI**: stock analog prototypes (LP/HP/BP/Notch/AP); decorator to wrap them as SciPy **TransferFunction** or callables.
- `src/sm_system/lse.py` — Helpers to build the LS system from spectra (indexing for 2nd/3rd‑order products, weighting) + **`least_squares`** solver.
- `src/sm_system/filter_design.py` — Minimum‑phase recovery and FIR design via frequency sampling.
- `src/sm_system/plots.py` — Bode and comparison plots for theory vs identification.
- `src/sm_system/config.py` — The **Configuration** dataclass (all run/probe/constraint/filter parameters).
- `src/sm_system/utils.py` — Numeric helpers (dB/levels, spacing, SNR/noise, matrix permutations, etc.).

---

## Typical workflow
1. **Configure** probe/FFT/noise/constraints in `Simulation().cfg`.
2. **Define** your SM model *(or attach a DUT)* via `SM_System` and `LTI` (or custom callables).
3. **Run** probes: `y = sim.run()`.
4. **Extract** steady‑state spectra at fundamental and intermod bins.
5. **Solve** for branch responses with `sm_system.lse` (optional constraints/weights).
6. **Validate & visualize** using Bode plots; optionally design FIRs matching the result.

---

## Notes & scope
- The identification procedure targets **steady-state** regimes.
- By default, probes and indexing are tuned for second- and third-order products; extending to higher orders requires adjusting the collision-avoidance rules and the least-squares (LS) assembly.
- There are ambiguities in dead-time and gain between the pre- and post-filters within a single system branch (see the plots in example.ipynb).
- A **uniqueness constraint** applies to ensure correct per-branch phase identification (phase overflow beyond [-pi, pi]).
- The cepstrum-based filter design needs improvement.
- Once an SM system is identified, its parameters can be used to design filters for **active distortion compensation**.
- The identification procedure has been **successfully applied in hardware setups**.

---

## Reference
**[BR75]** Stephen Baumgartner, Wilson Rugh, “Complete identification of a class of nonlinear systems from steady‑state frequency response,” *IEEE Trans. Circuits and Systems*, 22(9), pp. 753–759, 1975.

---

## License
See `LICENSE` in the repository.
