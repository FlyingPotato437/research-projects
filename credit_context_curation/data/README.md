# Data

This folder is intentionally empty. The notebook needs one file:

    data/lending_club.csv

It is roughly 1.6 GB and is distributed under Kaggle's terms, so it is not
redistributed in this repository.

## How to get it

1. Download the Lending Club dataset from Kaggle:
   https://www.kaggle.com/datasets/wordsforthewise/lending-club
2. Extract the accepted-loans CSV. It is the file containing the columns
   `issue_d` and `loan_status` (named `accepted_2007_to_2018Q4.csv`).
3. Rename it to `lending_club.csv` and place it in this folder.

## Using a copy you already have

If the file already lives somewhere else, point the notebook at it instead of
copying it, by setting an environment variable before starting Jupyter:

    export LENDING_CLUB_CSV=/absolute/path/to/accepted_2007_to_2018Q4.csv

## What runs without it

Sections 1 through 6 need this file. Section 7, the full-study results, reads
the small summary tables in `../stats/` and runs with or without it.
