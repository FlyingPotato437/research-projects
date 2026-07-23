# Context Curation for In-Context Tabular Foundation Models

A single, self-contained notebook that walks through an entire research study, written to be
readable by someone with no prior machine-learning background.

**The question.** New "in-context" foundation models for tables (TabPFN, TabFM) do not train the
usual way. You hand them a small table of labeled examples, a *context*, and they answer immediately,
but they can only read about a thousand rows. When the history holds far more than that, which rows
should the context contain, and does a smart choice matter when the economy has drifted between the
past and the present? We study this on real Lending Club loan data, predicting loan default.

## Contents

    credit_context_curation_explained.ipynb   the whole project, end to end, with rendered outputs
    stats/                                     small result tables for the at-scale study (Section 7)
    data/                                      where the raw loan file goes (see data/README.md)
    requirements.txt                           pinned package versions

## How to run it

1. Create an environment and install the dependencies:

       python3 -m venv .venv && source .venv/bin/activate
       pip install -r requirements.txt

2. Get the loan data. It is about 1.6 GB and licensed, so it is not stored here. Follow
   `data/README.md` to download `lending_club.csv` into `data/`, or point the notebook at an existing
   copy with `export LENDING_CLUB_CSV=/path/to/accepted_2007_to_2018Q4.csv`.

3. Launch the notebook and run the cells top to bottom:

       jupyter notebook credit_context_curation_explained.ipynb

All file paths in the notebook are relative to its folder, so it runs from wherever you clone it.

## What needs what

- **Sections 1 to 6** (loading, exploration, preprocessing, context strategies, model comparison)
  need `data/lending_club.csv`.
- **Section 6** also runs the live TabPFN model. That cell needs `tabpfn`, `torch`, and a GPU; if any
  are missing it skips itself automatically and the rest of the notebook still runs.
- **Section 7** (the full multi-seed, two-dataset, two-foundation-model results) reads only the small
  tables in `stats/` and always runs.

The notebook is committed with all cell outputs already rendered, so you can read the results without
running anything.
