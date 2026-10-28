# 16-beam Approximate-DVM TTD beamformer: reproduction and evaluation

This repo reproduces and checks Perera, Lingsch, Tuztas & Madanayake, "A Fast True Time-Delay Wideband
Multi-Beam Beamforming Algorithm Based on a 16-Beam Approximate-DVM", IEEE Access 13, 2025
(`refs/perera2025_advm.pdf`). It is for EE3140 at IIT Hyderabad. The results come from array-factor
simulations with ideal components only, so there is no full-wave modelling and no hardware.

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy matplotlib pytest
.venv/bin/python -m pytest -q          # gates G1-G5 (30 tests)
.venv/bin/python scripts/run_all.py    # all CSVs, figures and .tex files (~30 s)
```

## The idea in five steps

1. **The problem: squint.** A phase-shifter (DFT/Butler) beamformer applies fixed phases. A phase is
   only correct at one frequency, so in a wideband signal each frequency points somewhere different:
   sinθ = (2k'/N)(f_max/f). That's beam squint (orange curves in `F1`, tilted ridge in `F3`).
2. **The fix: true time delay (TTD).** Delays are applied instead of phases. In the frequency domain a
   delay of p·τ is a multiplication by α^p, with α = e^{-jωτ}. All N beams together form the
   *delay Vandermonde matrix* Ã[k,l] = α^{kl}. Its beams point the same way at every frequency
   (blue lines in `F1`). Done naively, it costs (N-1)² delay/gain blocks (`DIRECT`).
3. **Making it fast (`EXACT`, from the paper's ref. [6]).** Use kl = (k² + l² - (k-l)²)/2. This splits
   Ã into diagonal "chirp" delays around a Toeplitz matrix. The Toeplitz matrix sits inside a 2N-point
   circulant, and a circulant is diagonalized by FFTs:
   Ã = (1/M)·D̂ Jᵀ F̃* D̆ F̃ J D̂ with M = 2N. The FFTs need O(N log N) multipliers.
4. **The paper's trick (`ADVM`).** Replace the two 32-point FFTs with a *multiplierless approximate
   DFT* (F̂₃₂, Appendix). It is five sparse matrices S0…S4 that contain only 0, ±1, ±j, so they need
   only adds and 90° rotations. Larger N recurse down to F̂₃₂. This roughly halves the gain-delay count
   at N = 16 (130 → 62), but the result is no longer exactly Ã.
5. **What we found.**
   - The cost savings check out: our op counter reproduces the paper's Lemma 3.2, Table 1 and
     Table 3 exactly.
   - The accuracy cost is larger than the paper says. ADVM sidelobes are about 5 dB worse than EXACT
     (paper: 1–2 dB).
   - There is about 1–2° of residual squint, so ADVM fails our "squint-free" test (< 0.5°).
   - Our own extension (`ADVM-b`) rounds FFT twiddles to 2^b phases. It only becomes good enough
     at b = 5, which for N = 16 *is* the exact FFT, so it offers no cheaper squint-free design.

All numbers are in `results/SUMMARY.md`, and every resolved ambiguity is in `DECISIONS.md`.

## Map

| Path | What it does |
|---|---|
| `dvm/params.py` | All constants (N, f_max, bands, grids, thresholds) |
| `dvm/array.py` | ULA steering vectors, symmetric-beam pre-delay, array factor |
| `dvm/dvm_direct.py` | Brute-force Ã = [α^{kl}] (ground truth) |
| `dvm/dvm_exact.py` | Fast DVM as a list of sparse factors; radix-2 DIF FFT |
| `dvm/adft.py` | Appendix transcription: S0…S4 and F̂₃₂ (eq. 14) |
| `dvm/dvm_approx.py` | Paper ADVM (Algorithms 1–4) and our ADVM-b |
| `dvm/methods.py` | Beamformer matrix per method: PS / DIRECT / EXACT / ADVM / ADVM-b |
| `dvm/opcount.py` | Counts gains/delays/adders on the same factor lists that are executed |
| `dvm/metrics.py` | Peak angle, SLL, pattern error, relative spectral-norm error |
| `tests/test_g1..g5.py` | Gates: EXACT==DIRECT, ADFT transcription, Lemma 3.2 counts, physics, ADVM-b convergence |
| `scripts/run_all.py` | Computes all results CSVs, then calls `make_figs.py` and `make_tex.py` |
| `report/results_section.tex` | Auto-generated IEEE results section (numbers pulled from CSVs) |
| `report/algorithm.tex` | Algorithms 1–4 as `algorithm`/`algpseudocode` boxes |

The key design choice is that every algorithm is a list of sparse matrices. The same list is
multiplied to produce the beams and is scanned to count the operations, so the complexity numbers
always describe the code that actually runs.
