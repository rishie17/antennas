# Squint-free multi-beam antenna array: true-time-delay beamforming with a fast (approximate) DVM

EE3140 course project, IIT Hyderabad. We reproduce and evaluate
**Perera, Lingsch, Tuztas & Madanayake, "A Fast True Time-Delay Wideband Multi-Beam Beamforming
Algorithm Based on a 16-Beam Approximate-DVM", IEEE Access 13, 2025** (`refs/perera2025_advm.pdf`),
and connect it to what a physical antenna array gains from it.

Everything is a Python simulation (array factor, ideal components). No hardware or EM solver.

---

## 1. The project in one paragraph

A 16-antenna array forms 16 beams at once over 5–10 GHz. To aim a beam, each antenna's signal is
delayed by `l·d·sinθ/c`. **Phase shifters** fake that delay with a phase, which is only right at
one frequency, so the beams drift with frequency (**beam squint**). **True time delay (TTD)** fixes
this, but done directly it needs 225 delay/gain blocks for 16 beams. The paper **factorizes the 16×16
delay matrix** (FFT-like) so that the same beams need far fewer RF parts (30 delay lines instead of 225).
It then **approximates** the FFT stages so they need no multipliers at all (ADVM).

**What we found:**

| Claim | Result |
|---|---|
| Cost savings of the paper | ✅ reproduced exactly (Lemma 3.2, Table 1, Table 3, N = 16…1024) |
| TTD keeps every beam on target over 5–10 GHz | ✅ 0 dB gain loss; phase shifters are within 3 dB for only 41% of the band |
| ADVM accuracy ≈ exact (paper: 1–2 dB sidelobe penalty) | ❌ we get 5.3 dB worse sidelobes and 1.7° residual squint |
| Delay medium | stripline works; waveguide brings squint back (dispersion) |
| Paper appendix | typo in matrix S4, row 19 (fixed, see `DECISIONS.md` D1) |

## 2. Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy matplotlib pytest
.venv/bin/python -m pytest -q          # 44 tests, ~10 s
.venv/bin/python scripts/run_all.py    # regenerates every CSV, figure, .tex and the Overleaf zip (~40 s)
```

**Report:** upload `report/overleaf_bundle.zip` to Overleaf (New Project → Upload Project) and compile
`progress_report.tex`. Locally: `cd report && pdflatex progress_report` (twice).

## 3. Background you need (viva-level)

1. **ULA.** Element `l` sees a plane wave from angle θ with phase `2π f l d sinθ / c`. We use
   `d = λ/2 at f_max = 10 GHz`, so there are no grating lobes in the band.
2. **Phase vs delay.** A phase shifter set at `f0` points the beam at `sinθ' = (f0/f)·sinθ` at any
   other frequency. That's squint. A delay is right at every frequency.
3. **Many beams = a matrix.** With `α = e^{-jωτ}` (one unit of delay), beam `k` uses delays `k·l·τ`,
   so all beams together are the **delay Vandermonde matrix** `Ã[k,l] = α^{kl}`. At `f_max` it equals
   the DFT matrix, which is why an FFT/Butler beamformer works at only one frequency.
4. **Fast DVM (exact).** `kl = (k² + l² − (k−l)²)/2` splits Ã into *chirp delays × Toeplitz × chirp
   delays*. The Toeplitz part sits inside a 2N-point circulant, and the FFT diagonalizes the circulant:
   `Ã = (1/2N) · D̂ Jᵀ F* D̆ F J D̂`. Cost drops from O(N²) to O(N log N).
5. **ADVM (the paper).** Same formula, but the 32-point FFTs are replaced by a multiplierless
   approximation `F̂32 = S4·S3·S2·S1·S0` (entries only 0, ±1, ±j). It's cheaper, but no longer exact.
6. **Matrix → hardware.** Delay `α^p` → delay line `p·τ` · butterfly `[1 1; 1 −1]` → 180° hybrid ·
   `±j` block → 90° hybrid · row sum → combiners. Only the two `D̂` stages contain delay lines.

The code comments in `dvm/dvm_exact.py` walk through step 4 in detail. Start reading there.

## 4. Repo map

```
dvm/                    the library (all the physics and math)
  params.py             every constant: N, band, grids, pass/fail thresholds
  array.py              ULA steering vectors, beam pre-delay, array factor
  dvm_direct.py         brute-force Ã = [α^{kl}], the ground truth
  dvm_exact.py          fast exact DVM as a list of sparse factors, radix-2 FFT   ← start here
  adft.py               the paper's S0…S4 and F̂32, typed in from the appendix
  dvm_approx.py         ADVM (paper) and ADVM-b (our extension)
  methods.py            one entry point: beam_matrix(method, N, f) for PS/DIRECT/EXACT/ADVM/ADVM-b
  metrics.py            beam peak angle, sidelobe level, pattern error, matrix error
  opcount.py            counts gain/delay blocks and adders on the factor lists
  hardware.py           factor lists → RF parts list, delay-line lengths, waveguide dispersion
  io.py                 tiny CSV helpers
scripts/
  run_all.py            runs everything → results/*.csv, then make_figs.py and make_tex.py
  make_figs.py          CSVs → results/figs/F1…F8
  make_tex.py           CSVs → report/results_section.tex + results/SUMMARY.md (auto-generated)
tests/                  test_g1…g5, test_hardware: the checks that keep the numbers honest
report/                 progress_report.tex (main, hand-written), algorithm.tex, results_section.tex (generated)
results/                every number (CSV) and figure; SUMMARY.md has the headline tables
refs/                   the paper
DECISIONS.md            every ambiguity we had to resolve, one line each (D1…D31)
```

**How a run flows:**

```
params ─► methods.beam_matrix(method, N, f) ─► array.array_factor ─► metrics ─► results/*.csv
              │                                                                     │
              ├─ dvm_direct / dvm_exact / dvm_approx (lists of sparse factors)      ├─► make_figs ─► figs/
              └─ the same factor lists ─► opcount + hardware (costs, RF parts)      └─► make_tex  ─► report/
```

**Key design choice:** every algorithm is a *list of sparse matrices*. The same list is multiplied to
get the beams, counted to get the cost, and mapped to RF parts. So the cost numbers always describe
the network that was actually simulated.

## 5. The figures (what each one says)

| Fig | Shows | The one point |
|---|---|---|
| F1 | beam angle vs frequency, 7 beams | TTD lines are flat; phase-shifter (PS) curves drift = squint |
| F2 | paper Fig. 3 replication (16 patterns × 4 frequencies) | ADVM patterns are close to EXACT but not equal |
| F3 | heat map: direction × frequency for one beam | the PS beam curves off target; the EXACT beam is a vertical stripe |
| F4 | ADVM-b sweep over b (our extension) | only works at b = 5, which is just the exact FFT, so it's a null result |
| F5 | hardware blocks vs N (log scale) | direct ∝ N², fast DVM ∝ N log N |
| F6 | worst sidelobe vs frequency | EXACT −13.2 dB (uniform-array theory); ADVM ~5 dB worse |
| F7 | real delay lines instead of ideal ones | stripline = ideal; waveguide drifts (dispersion), WR-90 cuts off below 6.56 GHz |
| F8 | gain toward a beam's target vs frequency | TTD holds 0 dB; PS loses 13–30 dB over most of the band |

## 6. Methods compared

| Name | What it is |
|---|---|
| `PS` | phase-shifter / DFT (Butler) beamformer, fixed at f_max: the baseline that squints |
| `DIRECT` | TTD done brute force, N² blocks |
| `EXACT` | fast exact DVM (O(N log N)), identical beams to DIRECT |
| `ADVM` | the paper: EXACT with multiplierless approximate FFT stages |
| `ADVM-b` | ours: EXACT with FFT twiddles rounded to 2^b phases |

Pass/fail thresholds (fixed before running, never tuned): squint < 0.5°, pointing error < 0.5°,
sidelobe loss < 1 dB vs EXACT. They live in `dvm/params.py`.

## 7. Working on this as a team

- **Never hand-edit** `results/*` or `report/results_section.tex`. Change the code, then rerun `scripts/run_all.py`.
- Run `pytest -q` before you push. If a test fails, the numbers in the report are wrong.
- Work on a branch and open a pull request into `main`: `git checkout -b <your-name>/<topic>`.
- If you resolve an ambiguity in the paper, add one line to `DECISIONS.md`.
- Hand-written report text goes in `report/progress_report.tex`.

## 8. Limitations

Ideal lossless components, isotropic elements, no mutual coupling, array factor only. Hardware
area/power are estimates from component counts, not circuit results. The `D̆` stage is
frequency-dependent and was not designed as a circuit.
