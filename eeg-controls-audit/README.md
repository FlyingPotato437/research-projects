# Can you trust a brain-reading AI? Auditing the controls used to "prove" EEG decoders read the brain

A single, self-contained Jupyter notebook that walks through a complete small research project in
neurotechnology: **whether the "controls" scientists use to argue that a brain–computer-interface
decoder relies on real neural signal are themselves reliable.**

Perturbation "controls" — remove a brain rhythm, scramble channels, suppress a region — are widely
used as evidence that an EEG motor-imagery decoder depends on genuine brain activity. But a control
you never validated is not evidence. This project builds a controllable MI-EEG **simulator with known
ground truth**, measures each control's sensitivity and false-alarm rate against that truth across a
grid of conditions and **four decoder families** (band-power, CSP, a random-forest tree, and the
EEGNet neural network), applies the calibrated battery to **real** decoders, and finally confirms the
findings on **real brains** (eyes-open vs eyes-closed, where physiology fixes the answer).

**The headline:** the same controls, on the same data, give conclusions you should trust for one
decoder and throw out for another — a control's reliability **depends on the decoder**, and nobody was
checking which.

The notebook is written to be readable without prior neuroscience or machine-learning background, with
each concept explained as it appears. It has two halves:

- **Part I** — a hands-on, runnable walkthrough: what EEG is, how the data is preprocessed, a tiny
  brain simulator, several decoders, the controls, and the "does the control cry wolf?" experiment.
- **Part II** — the research itself: the contribution and novelty, the full-scale methods, the
  publication figures (operating characteristics, the calibrated real-decoder audit, and the
  real-brain validation), and an honest account of what the study does and does not prove.

## Run it

Fully self-contained — no data downloads, no internet, no special hardware:

```bash
pip install -r requirements.txt
jupyter notebook EEG_controls_explained_for_students.ipynb   # run top to bottom (~2 min on a laptop CPU)
```

Running it also writes publication-ready figures (vector PDF + 300-dpi PNG) into `paper_figures/`.

## Contents

| Path | What it is |
|---|---|
| `EEG_controls_explained_for_students.ipynb` | The notebook — read it, or run it to reproduce every figure |
| `paper_figures/` | Publication figures: Fig 1 simulator testbed · Fig 2 measured operating characteristics · Fig 3 calibrated real-decoder audit · Fig 4 real-brain (eyes-open/closed) validation · supporting accuracy |
| `build_student_notebook.py` | Regenerates the notebook (`python build_student_notebook.py`) |
| `requirements.txt` | The five libraries the notebook needs (+ `ipykernel`) |

## What the study finds

- **On the simulator (known truth):** simple decoders (band-power, tree) false-alarm on many controls —
  they "detect" dependence that provably isn't there — while EEGNet is specific. The tree, by only using
  the features it splits on, halves band-power's *spatial* false alarms, but is fooled the same on *band*
  controls, showing the failure is partly the features and partly the model type.
- **On real decoders:** the calibrated battery reaches opposite trust verdicts for EEGNet vs band-power
  on the *same* PhysioNet data.
- **On real brains:** in the eyes-open/closed task (where physiology tells us the true signal is occipital
  alpha), the simple decoders false-alarm exactly as the simulator predicted — the sim→real bridge.

## Honest limitation

Ground truth on the simulator is *operational*, not a real cortex, so how fully the measured reliabilities
transfer to every real setting is itself an open question. The eyes-open/closed result is real-brain
corroboration for the band/region controls, not a universal proof.
