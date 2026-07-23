# RadAlert-AT: Cross-Study Radiation-Exposure Triage

This project asks whether an *Arabidopsis* RNA-seq profile can identify radiation exposure when the
entire test study is unseen during training.

The topic was redesigned from exact dose regression after auditing the experimental design. Of 158
samples, 138 occur at either 0 or 100 Gy; only 20 represent intermediate doses. That distribution
supports exposed-versus-control triage much better than a precise calibration curve.

## Main result

Under six-fold leave-one-study-out evaluation, the stability ensemble obtains:

- **84.2% pooled accuracy**
- **82.4% balanced accuracy**
- **0.883 pooled ROC-AUC**
- **0.656 Matthews correlation coefficient**
- **89.0% sensitivity**
- **75.9% specificity**

Gene selection is fitted independently inside each training fold. No held-out-study labels are used
for feature selection, scaling, model fitting, or threshold choice.

## Start here

Open `RadDose_Full_Walkthrough.ipynb`. The fully executed notebook contains:

1. a target-design audit explaining the change from regression to exposure triage;
2. the six-study data and preprocessing contract;
3. leakage-safe leave-one-study-out evaluation;
4. literature-panel, linear, kernel, and tree baselines;
5. a 50/100-gene stability ensemble;
6. accuracy, balanced accuracy, AUC, average precision, MCC, sensitivity, and specificity;
7. pooled and per-study confusion/ROC analysis;
8. performance at 0, 10, 40, 80, and 100 Gy;
9. study-level bootstrap uncertainty;
10. stable-gene and DNA-damage-response interpretation;
11. exact dose regression retained as a transparent secondary negative result.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook RadDose_Full_Walkthrough.ipynb
```

The notebook runs on CPU in roughly half a minute in the verified environment.

## Included data

`raddose_data/` contains the harmonized 158 × 4,002 expression matrix, sample manifest, two raw-study
examples used for teaching figures, and saved biological-interpretation tables.

Samples originate from NASA OSDR studies OSD-498, OSD-502, OSD-508, OSD-510, OSD-658, and OSD-782.

## Claim boundary

This is an exploratory plant-biology exposure classifier. It is not a human dosimeter, a deployed
safety system, or evidence that the selected genes are specific to radiation rather than other
stressors. With only six external studies, the most valuable next step is a frozen-model test on an
independent seventh study containing multiple doses and non-radiation stress controls.
