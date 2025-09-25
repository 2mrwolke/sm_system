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
This repo is pure Python. Typical dependencies include `numpy`, `scipy`, `matplotlib`, `tqdm`, `pytz`.
```bash
# Option 1: add the module folder to your path
export PYTHONPATH="$PYTHONPATH:/path/to/sm_system/module"

# Option 2: vendor it in your project or turn it into a package
# (pip/pyproject not included in this repo snapshot)
```

### Minimal example (simulation + identification skeleton)

For an end‑to‑end walk‑through, see **`example.ipynb`** (explains probe generation, indexing of intermod bins, and the LS setup).

---

## Modules (what lives where)
- `module/sm.py` — **`SM_System`**: compose per‑order branches (`add`, `merge`, callable system).
- `module/sim.py` — **`Simulation`**: orchestrates configuration, probe generation, runs the system/DUT.
- `module/probe.py` — **HarmonicProbeIterator**: creates frequency‑pair batches that avoid collisions up to a chosen order.
- `module/lti.py` — **LTI**: stock analog prototypes (LP/HP/BP/Notch/AP); decorator to wrap them as SciPy **TransferFunction** or callables.
- `module/lse.py` — Helpers to build the LS system from spectra (indexing for 2nd/3rd‑order products, weighting) + **`least_squares`** solver.
- `module/filter_design.py` — Minimum‑phase recovery and FIR design via frequency sampling.
- `module/plots.py` — Bode and comparison plots for theory vs identification.
- `module/config.py` — The **Configuration** dataclass (all run/probe/constraint/filter parameters).
- `module/utils.py` — Numeric helpers (dB/levels, spacing, SNR/noise, matrix permutations, etc.).

---

## Typical workflow
1. **Configure** probe/FFT/noise/constraints in `Simulation().cfg`.
2. **Define** your SM model *(or attach a DUT)* via `SM_System` and `LTI` (or custom callables).
3. **Run** probes: `y = sim.run()`.
4. **Extract** steady‑state spectra at fundamental and intermod bins.
5. **Solve** for branch responses with `module.lse` (optional constraints/weights).
6. **Validate & visualize** using Bode plots; optionally design FIRs matching the result.

---

## Notes & scope
- The identification procedure targets **steady‑state** regimes as in Baumgartner & Rugh (1975).
- Probes and indexing are tuned for **2nd/3rd‑order** products by default; extending to higher orders requires adjusting the collision‑avoidance rules and LS assembly.
- There are ambiguities in dead time and gain between the pre- and post-filters of a single system branch (see the plots in example.ipynb).
- The cepstrum-based filter design needs improvement.
- Once an SM system is identified, its parameters can be used to design filters for active distortion compensation.
- The identification procedure has been successfully applied in hardware setups.

---

## Hardware Setup

#### TODO - Add images of setup and results

---

## Reference
**[BR75]** Stephen Baumgartner, Wilson Rugh, “Complete identification of a class of nonlinear systems from steady‑state frequency response,” *IEEE Trans. Circuits and Systems*, 22(9), pp. 753–759, 1975.

---

## License
See `LICENSE` in the repository.
