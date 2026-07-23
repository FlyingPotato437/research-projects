import fs from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");

function lines(text) {
  return text.trim().split("\n").map((line, i, a) => i < a.length - 1 ? `${line}\n` : line);
}

function markdown(id, text) {
  return { cell_type: "markdown", id, metadata: {}, source: lines(text) };
}

function findCell(nb, prefix) {
  const index = nb.cells.findIndex(cell => cell.source.join("").startsWith(prefix));
  if (index < 0) throw new Error(`Could not find cell beginning: ${prefix}`);
  return index;
}

function insertAfter(nb, prefix, ...cells) {
  nb.cells.splice(findCell(nb, prefix) + 1, 0, ...cells);
}

function insertBefore(nb, prefix, ...cells) {
  nb.cells.splice(findCell(nb, prefix), 0, ...cells);
}

function replaceMarkdown(nb, prefix, text) {
  const cell = nb.cells[findCell(nb, prefix)];
  if (cell.cell_type !== "markdown") throw new Error(`Expected markdown: ${prefix}`);
  cell.source = lines(text);
}

function replaceCode(nb, prefix, text, clearOutput = true) {
  const cell = nb.cells[findCell(nb, prefix)];
  if (cell.cell_type !== "code") throw new Error(`Expected code: ${prefix}`);
  cell.source = lines(text);
  if (clearOutput) {
    cell.execution_count = null;
    cell.outputs = [];
  }
}

function updateNotebook(relativePath, edit) {
  const file = path.join(root, relativePath);
  const nb = JSON.parse(fs.readFileSync(file, "utf8"));
  edit(nb);
  fs.writeFileSync(file, `${JSON.stringify(nb, null, 1)}\n`);
}

const evidenceGuide = (domainUnit, endpoint) => `
## How to read the evidence

This notebook separates four kinds of statement:

| label | meaning in this notebook |
|---|---|
| **Definition** | terminology or an equation; no empirical claim is being made |
| **Observation** | a pattern in the displayed sample or figure |
| **Result** | a quantity produced by the stated evaluation protocol |
| **Interpretation** | a plausible explanation that remains open to alternative causes |

The independent evaluation unit is **${domainUnit}**, and the principal endpoint is **${endpoint}**.
Comparisons are meaningful only when models use the same split, features, and endpoint. Decimal places
are shown for reproducibility, not as a claim of equivalent precision.`;

updateNotebook("Heart Disease Cross Hospital Research.ipynb", nb => {
  replaceMarkdown(nb, "# Heart Disease Across Hospitals", `
# Cross-Hospital Generalization of Heart-Disease Classifiers

> **Educational research notebook — not a diagnostic device.** The models below are evaluated on a
> historical benchmark and must not be used for clinical decisions.

**Abstract—** A model can perform well under internal validation yet fail when moved to a hospital
with different patients, measurements, and missing-data patterns. This notebook compares logistic
regression, random forests, and a small neural network on four UCI heart-disease cohorts. We first
estimate internal performance on Cleveland using stratified five-fold cross-validation. We then
perform directed source-to-target hospital transfer using the nine variables that are actually
observed across all sites. Preprocessing is fitted inside each training fold or on the source
hospital only. The study emphasizes discrimination (ROC-AUC), transfer gaps, and limitations caused
by small, imbalanced cohorts. Its purpose is to teach sound external-validation reasoning, not to
claim clinical readiness.

**Index Terms—** clinical machine learning, dataset shift, external validation, heart disease,
ROC-AUC, reproducibility.

### Research questions

1. How accurately do the three model families discriminate disease within Cleveland?
2. How much performance changes when the training and test hospitals differ?
3. Does model complexity improve internal fit at the cost of transportability?

### Learning outcomes

By the end, you should be able to identify disguised missing values, keep preprocessing inside the
training boundary, distinguish internal from external validation, interpret a transfer matrix, and
state conclusions at the strength supported by the data.

### Study map

| stage | data boundary | purpose |
|---|---|---|
| Part I | Cleveland, five held-out folds | model development and internal validation |
| Part II | one source hospital → one different target hospital | external transport test |
| Synthesis | model-level macro averages across directed transfers | compare robustness, with caveats |
`);
  insertAfter(nb, "# Cross-Hospital Generalization", markdown("ieee-heart-evidence",
    evidenceGuide("a patient within an internal fold, and a target hospital in the transfer analysis",
      "ROC-AUC; accuracy and F1 are supporting endpoints")));
  replaceMarkdown(nb, "Our model is strong on Cleveland.", `
The Cleveland analysis estimates **internal** discrimination. It does not answer the deployment
question: what happens when the case mix, measurement process, and missingness pattern change?

The travel test therefore uses only the **nine measurements with genuine coverage across all four
hospitals**. In particular, Switzerland records cholesterol as zero for every patient, so
cholesterol is not treated as a portable feature. This is a stricter but more defensible comparison
than filling an entirely absent source variable with information pooled from other hospitals.`);
  replaceCode(nb, "FEATURES_SHARED=", `
FEATURES_SHARED=["age","sex","cp","trestbps","fbs","restecg","thalach","exang","oldpeak"]
print("Portable measurements used for the travel test:", len(FEATURES_SHARED))

def fit_preprocessor(train_df):
    """Fit imputation and scaling on the training rows only."""
    med = train_df[FEATURES_SHARED].median()
    if med.isna().any():
        missing = med.index[med.isna()].tolist()
        raise ValueError(f"Training source has no observations for: {missing}")
    filled = train_df[FEATURES_SHARED].fillna(med)
    scaler = StandardScaler().fit(filled)
    def transform(df):
        return scaler.transform(df[FEATURES_SHARED].fillna(med))
    return transform`);
  replaceMarkdown(nb, "We train each model on **all** of one hospital", `
For every off-diagonal source→target pair, preprocessing and model fitting use the source hospital
only; the target hospital is untouched until scoring. On the diagonal, preprocessing is re-fitted
inside every cross-validation fold. This distinction prevents test-fold statistics from leaking
into the model. Because target hospitals vary greatly in size and prevalence, each directed pair is
shown rather than hidden behind one pooled score.`);
  replaceCode(nb, "import time\ndef safe_auc", `
import time
def safe_auc(yy,p): return roc_auc_score(yy,p) if len(np.unique(yy))>1 else np.nan
def fit_lr(X,yv): return LogisticRegression(C=1.0,max_iter=2000,class_weight="balanced",random_state=SEED).fit(X,yv)
def fit_rf2(X,yv): return RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=SEED,n_jobs=-1).fit(X,yv)
def fit_mlp2(X,yv):
    tf.keras.utils.set_random_seed(SEED); m=build_mlp(X.shape[1])
    n0,n1=int((yv==0).sum()),int((yv==1).sum()); cw={0:len(yv)/(2*n0),1:len(yv)/(2*n1)} if min(n0,n1)>0 else None
    es=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=15,restore_best_weights=True)
    if min(n0,n1)>=5: m.fit(X,yv,validation_split=0.2,epochs=200,batch_size=16,verbose=0,callbacks=[es],class_weight=cw)
    else: m.fit(X,yv,epochs=120,batch_size=16,verbose=0,class_weight=cw)
    return m

models=["LogReg","RandomForest","MLP"]; AUC={m:np.full((4,4),np.nan) for m in models}; records=[]
t0=time.time()
for i,tr in enumerate(sites):
    source=data[tr]
    source_prep=fit_preprocessor(source)
    Xtr,ytr=source_prep(source),source["target"].values
    lr,rf,mm=fit_lr(Xtr,ytr),fit_rf2(Xtr,ytr),fit_mlp2(Xtr,ytr)
    for j,te in enumerate(sites):
        if i==j:
            skf2=StratifiedKFold(5,shuffle=True,random_state=SEED)
            oof_site={m:np.full(len(source),np.nan) for m in models}
            for a,b in skf2.split(source,ytr):
                fold_prep=fit_preprocessor(source.iloc[a])
                Xa,Xb=fold_prep(source.iloc[a]),fold_prep(source.iloc[b])
                oof_site["LogReg"][b]=fit_lr(Xa,ytr[a]).predict_proba(Xb)[:,1]
                oof_site["RandomForest"][b]=fit_rf2(Xa,ytr[a]).predict_proba(Xb)[:,1]
                oof_site["MLP"][b]=fit_mlp2(Xa,ytr[a]).predict(Xb,verbose=0).ravel()
            for m in models: AUC[m][i,j]=safe_auc(ytr,oof_site[m])
            regime="same hospital"
        else:
            Xte,yte=source_prep(data[te]),data[te]["target"].values
            AUC["LogReg"][i,j]=safe_auc(yte,lr.predict_proba(Xte)[:,1])
            AUC["RandomForest"][i,j]=safe_auc(yte,rf.predict_proba(Xte)[:,1])
            AUC["MLP"][i,j]=safe_auc(yte,mm.predict(Xte,verbose=0).ravel())
            regime="new hospital"
        for m in models:
            records.append({"train_site":tr,"test_site":te,"model":m,
                            "regime":regime,"ROC_AUC":AUC[m][i,j]})
n_cross=sum(1 for r in records if r["regime"]=="new hospital")//len(models)
print(f"Tested every hospital pair in {time.time()-t0:.0f}s ({n_cross} new-hospital tests per model).")`);
  insertBefore(nb, "## 📋 Discussion summary", markdown("ieee-heart-validity", `
## Validity checks and interpretation rules

- **No preprocessing leakage:** medians and scaling parameters are learned only from training rows.
- **No pseudo-shared cholesterol:** the all-zero Swiss field is excluded from the transport feature set.
- **Different estimands:** Part I (13 features) and Part II (9 features) answer different questions and
  must not be compared as if only the hospital changed.
- **Hospital is the external unit:** twelve directed transfers are informative examples, not twelve
  independent replications of clinical deployment.
- **ROC-AUC is not calibration or utility:** a deployable system would also require calibration,
  threshold-specific sensitivity/specificity, subgroup analysis, prospective validation, and a
  decision-curve or cost analysis.
`));
  insertAfter(nb, "### What we found", markdown("ieee-heart-refs", `
## References

[1] R. Detrano *et al.*, “International application of a new probability algorithm for the
diagnosis of coronary artery disease,” *American Journal of Cardiology*, vol. 64, no. 5, 1989.

[2] G. S. Collins *et al.*, “Transparent Reporting of a multivariable prediction model for
Individual Prognosis Or Diagnosis (TRIPOD),” *Annals of Internal Medicine*, vol. 162, 2015.

[3] E. W. Steyerberg and F. E. Harrell, “Prediction models need appropriate internal, internal–external,
and external validation,” *Journal of Clinical Epidemiology*, vol. 69, 2016.

### Reproducibility record

The notebook fixes random seeds, records the data source, uses explicit feature lists, fits
preprocessing inside training boundaries, exports pairwise scores, and retains every directed
hospital result. A rerun may vary slightly for the neural network across hardware/library versions;
the qualitative conclusion should be judged from the full transfer matrix, not one rounded average.
`));
});

updateNotebook("clip_forgetting_forecasting/CLIP_Forgetting_Forecasting.ipynb", nb => {
  replaceMarkdown(nb, "# Forecasting Catastrophic Forgetting in CLIP", `
# Early Forecasting of Catastrophic Forgetting in CLIP Fine-Tuning

**Abstract—** Fine-tuning a pretrained vision-language model can improve a target task while reducing
its zero-shot performance elsewhere. This notebook studies whether final forgetting can be forecast
from four pre-training descriptors and the first five fine-tuning epochs. The experiment contains 48
runs formed by 16 datasets and three learning rates. Evaluation holds out an entire dataset, including
all of its learning rates, so the forecasting model cannot memorize dataset identity. Static features
provide moderate predictive value; adding early dynamics substantially improves the cached
leave-one-dataset-out results. The study is deliberately framed as a small-sample forecasting result:
one CLIP architecture, one optimizer, one run per configuration, and only 16 independent dataset
groups. It supports a promising association, not a universal law of forgetting.

**Index Terms—** catastrophic forgetting, CLIP, fine-tuning dynamics, grouped cross-validation,
multimodal learning, transfer learning.

### Research questions and contributions

1. Which dataset and optimization factors are associated with final zero-shot forgetting?
2. Does a five-epoch warm-up improve forecasts for a dataset excluded from training?
3. Do simple tabular regressors outperform sequence models in the 48-run regime?
4. The notebook contributes an end-to-end, cached, reproducible comparison with group-wise
   evaluation and an explicit static-only ablation.
`);
  insertAfter(nb, "# Early Forecasting", markdown("ieee-clip-evidence",
    evidenceGuide("a dataset (not an individual learning-rate run)",
      "leave-one-dataset-out R²; MAE and Spearman correlation are supporting endpoints")));
  replaceMarkdown(nb, "### 8.1 How we test honestly", `
### 8.1 Grouped evaluation: leave one dataset out

All three learning-rate runs from the same dataset share static features and are statistically
dependent. Random row splitting would therefore leak dataset identity. We instead hold out one
dataset at a time. Every transformation—including the LSTM's sequence and static-feature
normalization—is fitted on the 15 training datasets inside that fold. This protocol estimates
transfer to another dataset drawn from a similar benchmark collection; it does not estimate transfer
to a new architecture, optimizer, or fine-tuning recipe.`);
  replaceMarkdown(nb, "### 9.4 Which inputs did the model rely on?", `
### 9.4 Descriptive feature reliance (not a causal explanation)

The following random-forest importance is fitted to the full 48-row table. It is useful for seeing
which variables the fitted forest uses, but correlated predictors can divide importance arbitrarily,
and in-sample impurity importance is not a causal attribution. Treat the chart as a hypothesis
generator. A stronger follow-up would compute fold-wise permutation importance and report its
variation across held-out datasets.`);
  insertBefore(nb, "## Part 9.6 Reproducing", markdown("ieee-clip-uncertainty", `
## Part 9.6 Uncertainty and robustness audit

The three learning rates within a dataset are not independent replicates. Accordingly, the effective
sample size for generalization is 16 dataset groups, not 48 rows. The large warm-up gain is the
principal result, while its second or third decimal place is not stable evidence. A publication-grade
extension should add:

1. multiple fine-tuning seeds for every configuration;
2. nested, group-aware model selection if hyperparameters are tuned;
3. a dataset-level bootstrap or repeated benchmark resampling interval;
4. architecture, optimizer, and prompt-template replications;
5. a persistence baseline such as “final forgetting equals forgetting at epoch 5.”

That final baseline is especially important: it separates sophisticated forecasting from simple
curve continuation.
`));
  replaceMarkdown(nb, "## Part 10. Conclusions", `
## Part 10. Conclusions

**Supported findings**

1. Final forgetting varies markedly across datasets and learning rates in this experiment.
2. Under leave-one-dataset-out evaluation, early training dynamics add substantial forecast signal
   beyond the static descriptors used here.
3. Simple regressors are more reliable than the MLP and LSTM in this small grouped sample.

**Claim boundary**

The study contains 16 independent dataset groups, one CLIP architecture, one optimizer, and one run
per configuration. It therefore establishes an internally consistent benchmark result—not a
population-level guarantee that five epochs forecast forgetting for arbitrary foundation models.
The next decisive experiment is a multi-seed, multi-architecture replication against a persistence
baseline.

## References

[1] A. Radford *et al.*, “Learning Transferable Visual Models From Natural Language Supervision,”
*Proceedings of ICML*, 2021.

[2] J. Kirkpatrick *et al.*, “Overcoming catastrophic forgetting in neural networks,”
*Proceedings of the National Academy of Sciences*, vol. 114, no. 13, 2017.

[3] S. Kornblith, J. Shlens, and Q. V. Le, “Do Better ImageNet Models Transfer Better?”
*Proceedings of CVPR*, 2019.

### Reproducibility record

The repository includes the experiment generator, modeling helper, cached dynamics, explicit group
labels, fixed model seeds where supported, and the exact command for rebuilding the 48-run campaign.
The LSTM helper fits normalization inside each held-out-dataset fold.
`);
});

updateNotebook("credit_context_curation/credit_context_curation_explained.ipynb", nb => {
  replaceMarkdown(nb, "# Choosing What an AI Model Should Study", `
# Context Curation for Credit-Risk Prediction Under Temporal Shift

**Abstract—** In-context tabular foundation models can condition on only a limited set of labeled
rows. This notebook asks which historical loans should form that context when the economic
environment changes over time. Six selection strategies are compared on temporally separated credit
cohorts using discrimination and calibration metrics, with conventional learners trained on the same
contexts and tuned XGBoost trained on the full pool as controls. The principal result is data
efficiency rather than universal superiority: sensible curated contexts can approach a strong
full-data baseline, while a high-confidence heuristic can reverse the predictive signal. The study
also exposes deployment constraints, including aggregate target-period information used by one
strategy and the limited number of dataset/split replications.

**Index Terms—** context curation, credit risk, dataset shift, in-context learning, TabPFN,
temporal validation.

### Research questions

1. Does context selection change performance when context size is fixed?
2. Which strategies remain useful across time splits and foundation-model families?
3. Can a 1,024-row context approach a tuned learner trained on the full historical pool?

> **Responsible-use boundary.** This is a methodological benchmark, not a lending policy. It does
> not evaluate fairness, adverse-action explanations, regulatory compliance, or decision utility.
`);
  insertAfter(nb, "# Context Curation", markdown("ieee-credit-evidence",
    evidenceGuide("a dataset/time-split/seed combination",
      "ROC-AUC; average precision, Brier score, and calibration error are supporting endpoints")));
  replaceMarkdown(nb, "### Do the two foundation models agree?", `
### Cross-model replication

The following comparison asks whether the direction of context effects repeats for TabPFN and
TabFM. Agreement strengthens robustness, but correlated scores on the same contexts do **not** prove
that a behavior holds for in-context learning in general. Logistic regression and XGBoost trained on
the identical context help distinguish a model-family effect from a property of the selected rows.`);
  replaceMarkdown(nb, "### The honest headline: parity, not magic", `
### Full-pool comparison: data efficiency, with selection uncertainty

The main reference is tuned XGBoost trained on all available pool rows. The plotted gap compares
that baseline with the best observed curated foundation-model context. Because the best strategy is
selected after comparing candidates, its estimate can be optimistic unless strategy selection is
nested inside an independent evaluation. The intervals below quantify recorded seed variation; they
should not be read as universal confidence intervals over future economies. An interval spanning zero
is evidence of **inconclusive difference at this resolution**, not proof that the methods are
identical.`);
  replaceMarkdown(nb, "## 8. What we learned", `
## 8. Conclusions, validity, and deployment boundary

**Supported findings**

1. Context composition changes performance even at fixed context size.
2. The high-confidence rule can select a subgroup with a reversed feature–outcome relationship,
   degrading multiple learner families.
3. TabPFN and TabFM show similar relative behavior on the recorded contexts.
4. The best 1,024-row contexts are competitive with the full-pool tuned XGBoost in several recorded
   splits; this is evidence of data efficiency, not guaranteed superiority.

**Threats to validity**

- The best-of-six comparison has selection optimism unless strategy choice is independently nested.
- **economically_similar** uses aggregate target-period information and is transductive; a live system
  would need those aggregates without labels and without future leakage.
- Dataset/split counts, rather than the number of loans, govern the breadth of the generalization claim.
- AUC does not assess fairness, probability calibration at an operating threshold, lending utility,
  or compliance requirements.
- The raw Lending Club file is externally licensed, so Sections 1–6 require a separate download;
  the committed summary tables support audit of Section 7.

## References

[1] N. Hollmann *et al.*, “TabPFN: A Transformer That Solves Small Tabular Classification Problems
in a Second,” *International Conference on Learning Representations*, 2023.

[2] T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” *Proceedings of KDD*, 2016.

[3] C. Guo *et al.*, “On Calibration of Modern Neural Networks,” *Proceedings of ICML*, 2017.

### Reproducibility record

The notebook records temporal pools and test periods, context sizes, selection algorithms, metrics,
seeds in the saved aggregates, identical-context baselines, and full-pool comparisons. To extend the
work, pre-register one context strategy on development splits and evaluate it once on a later,
untouched period.
`);
});

updateNotebook("eeg-controls-audit/EEG_controls_explained_for_students.ipynb", nb => {
  replaceMarkdown(nb, "# Can you trust a brain-reading AI?", `
# Calibrating Perturbation Controls for EEG Decoder Audits

**Abstract—** Perturbation controls are often used to argue that an EEG decoder relies on a specific
frequency band or scalp region. Such an argument is valid only if the control has known sensitivity
and false-alarm behavior for the decoder being audited. This notebook introduces a simulator with
operational ground truth, calibrates several band and spatial controls across decoder families,
applies the calibrated battery to real decoders, and compares the simulator’s warning with an
eyes-open/eyes-closed task whose occipital-alpha physiology is known. The key finding is conditional:
control reliability depends on the feature representation and decoder. Simulation supplies
identifiability; real data supplies corroboration, not universal ground truth.

**Index Terms—** brain–computer interfaces, EEG, explainability, perturbation tests, simulation,
validation.

### Research questions

1. When a known signal is present, how often does each control detect it?
2. When that signal is absent, how often does the control produce a false explanation?
3. Do calibration results change with the decoder and feature representation?
4. Which simulator findings are corroborated by a real task with a well-established physiological signal?

### Two reading tracks

- **Part I** is the teachable implementation: signal, simulator, decoders, perturbations, and outcomes.
- **Part II** is the research report: contribution, protocol, operating characteristics, real-data
  audit, limitations, and implications.
`);
  insertAfter(nb, "# Calibrating Perturbation", markdown("ieee-eeg-evidence",
    evidenceGuide("a simulated condition/seed/decoder combination; subjects are the unit in real-data validation",
      "control sensitivity and false-alarm rate; decoder accuracy is a prerequisite, not the audit endpoint")));
  replaceMarkdown(nb, "## 14. What is new here", `
## 14. Contribution and scope

The contribution is an **instrument-calibration view** of perturbation explanations. A control is
not assumed trustworthy because it is intuitive; it is measured under conditions where dependence
is known. The contribution is methodological and empirical within the tested simulator, decoder
families, and tasks. It does not establish that the simulator is a complete model of cortical EEG,
nor that one calibration table transfers unchanged to every recording pipeline.`);
  insertBefore(nb, "## 20. Limitations", markdown("ieee-eeg-validity", `
## 20. Validity matrix

| source of evidence | what it can establish | what it cannot establish |
|---|---|---|
| simulator | sensitivity and false alarms under planted operational truth | complete biological realism |
| motor-imagery decoder audit | whether conclusions survive calibrated controls on the recorded pipeline | latent neural ground truth |
| eyes-open/closed data | sim-to-real corroboration for occipital alpha | universal validity for every task or decoder |

Potential confounds include preprocessing order, volume conduction, spectral leakage, spatially
correlated noise, finite trial counts, and reuse of hyperparameters across conditions. The correct
unit of inference should remain explicit when confidence intervals are added: trials estimate a
subject-specific decoder, while subjects support population claims.
`));
  insertAfter(nb, "## 21. Recap", markdown("ieee-eeg-refs", `
## References

[1] H. Ramoser, J. Müller-Gerking, and G. Pfurtscheller, “Optimal spatial filtering of single trial
EEG during imagined hand movement,” *IEEE Transactions on Rehabilitation Engineering*, vol. 8,
no. 4, 2000.

[2] V. J. Lawhern *et al.*, “EEGNet: A compact convolutional neural network for EEG-based
brain–computer interfaces,” *Journal of Neural Engineering*, vol. 15, no. 5, 2018.

[3] S. S. Sanei and J. A. Chambers, *EEG Signal Processing*. Wiley, 2007.

### Reproducibility record

The notebook fixes simulation seeds, defines planted truth before evaluating controls, reports catch
and false-alarm regimes, keeps decoder families separate, and writes publication figures in both
vector and high-resolution raster formats. A stronger confirmatory study would add subject-level
intervals, preregistered thresholds, and an independent dataset.
`));
});

updateNotebook("exoplanet-model-comparison/Exoplanet_Model_Comparison.ipynb", nb => {
  replaceMarkdown(nb, "# Comparing Machine-Learning Models for Exoplanet Detection", `
# Accuracy–Interpretability Trade-offs in Exoplanet Transit Modeling

**Abstract—** This notebook compares seven machine-learning families on simulated transit detection
and radius-ratio estimation, then studies a label-free physics-informed KAN and a small real Kepler
sample. Performance is evaluated on a held-out synthetic set. “Interpretability” is operationalized
as the fidelity of an additive surrogate, which measures one specific property rather than all forms
of explanation quality. The physics-informed model reconstructs light curves without radius labels,
but benefits from a simulator closely matched to its forward model. Real-data results are reported as
a limited transfer demonstration; residual-based subset filtering is exploratory and cannot replace
performance on the full, pre-specified sample.

**Index Terms—** exoplanets, interpretable machine learning, Kolmogorov–Arnold networks,
physics-informed learning, transit photometry.

### Research questions

1. Which model families best detect synthetic transits and estimate $R_p/R_*$?
2. What accuracy is exchanged for the chosen additive-surrogate fidelity?
3. Can a differentiable transit model learn physical parameters without radius labels?
4. How far do simulation-trained estimates transfer to a small Kepler sample?
`);
  insertAfter(nb, "# Accuracy–Interpretability", markdown("ieee-exo-evidence",
    evidenceGuide("a simulated light curve; a planet in the real-data demonstration",
      "ROC-AUC for detection and MAE in radius ratio for characterization")));
  replaceMarkdown(nb, "## 8. Measuring interpretability objectively", `
## 8. Operationalizing one dimension of interpretability

Interpretability has no single universal scalar definition. Here we measure **additive-surrogate
fidelity**: how closely a sum of per-time-point functions reproduces a model's output on an
independent probe set. A high score means the model behaves approximately additively over that probe
distribution. It does not by itself establish causal correctness, human usefulness, stability, or
physical validity. The score is nevertheless valuable because it is explicit, reproducible, and
applied consistently to every model.`);
  replaceMarkdown(nb, "### 11.2 Recovering real planet sizes", `
### 11.2 Real-data transfer demonstration

The model is applied to 12 embedded, confirmed Kepler planets. The **full 12-planet table is the
primary report**. The plot additionally marks the lowest-residual 75% as “well fit”; because that
threshold is chosen from the same sample after observing reconstruction residuals, its MAE is an
exploratory diagnostic and is expected to look better than an unfiltered result. It must not be
presented as the pre-specified real-data accuracy. Residual screening is useful for identifying model
mismatch, but a confirmatory study would set the quality threshold in advance and report both
accepted and rejected cases.`);
  replaceMarkdown(nb, "## 12. Conclusions", `
## 12. Conclusions and claim boundaries

**Supported findings**

- Several model families perform strongly on held-out curves from the stated synthetic generator.
- Accuracy and additive-surrogate fidelity are not identical objectives; the Pareto view makes the
  chosen trade-off visible.
- A differentiable transit model can train a KAN by reconstruction without radius labels in this
  matched-simulation setting.
- The embedded Kepler sample demonstrates qualitative sim-to-real transfer and exposes cases of
  model mismatch.

**Threats to validity**

- Synthetic train and test data share a generator, parameter ranges, and noise assumptions.
- The additive score captures only one dimension of interpretability.
- The physics-informed learner has a strong matched-model advantage.
- The 12-planet real sample is small and not a population estimate.
- The residual-filtered subset is post hoc; the full sample is the primary real-data evidence.
- Training stochasticity is not summarized across repeated seeds.

## References

[1] K. Mandel and E. Agol, “Analytic Light Curves for Planetary Transit Searches,”
*The Astrophysical Journal Letters*, vol. 580, no. 2, 2002.

[2] W. J. Borucki *et al.*, “Kepler Planet-Detection Mission: Introduction and First Results,”
*Science*, vol. 327, no. 5968, 2010.

[3] Z. Liu *et al.*, “KAN: Kolmogorov–Arnold Networks,” arXiv:2404.19756, 2024.

### Reproducibility record

The notebook fixes data-generation seeds, separates train/test simulations, validates the
differentiable forward model against **batman**, embeds the real sample, and exposes every model
comparison. A publication-grade extension should repeat training seeds, pre-register a real-data
quality rule, report the full-sample MAE with uncertainty, and test mismatched noise plus TESS data.
`);
});

updateNotebook("raddose-phytodosimetry/RadDose_Full_Walkthrough.ipynb", nb => {
  replaceMarkdown(nb, "# RadDose-AT", `
# Cross-Study Phytodosimetry from *Arabidopsis* Gene Expression

**Abstract—** This notebook evaluates whether RNA-seq profiles can recover absorbed radiation dose
across six independent *Arabidopsis* studies. The analysis contrasts random within-study splitting
with leave-one-study-out (LOSO) validation, compares ten model families, and inspects compact gene
panels and external chemical-damage concordance. The central result is modest but useful: expression
preserves dose ranking across studies better than simple baselines, while absolute dose error remains
large. Because the distributed matrix contains a harmonized, preselected gene set, LOSO evaluates
model transfer conditional on that representation; it is not a fully nested raw-count pipeline.
Biological annotations and external concordance support plausibility but do not establish causality.

**Index Terms—** bioinformatics, domain generalization, gene expression, ionizing radiation,
leave-one-study-out validation, phytodosimetry.

### Research questions

1. How well can dose be ranked and estimated in a study excluded from model fitting?
2. Which model families retain performance across experimental batches?
3. Can a compact gene panel preserve cross-study signal?
4. Do selected genes align with known DNA-damage biology and an independent perturbation dataset?
`);
  insertAfter(nb, "# Cross-Study Phytodosimetry", markdown("ieee-rad-evidence",
    evidenceGuide("a held-out study for cross-study claims; samples are dependent within studies",
      "LOSO MAE in Gray and Spearman rank correlation")));
  replaceMarkdown(nb, "The best model (gradient-boosted trees)", `
Gradient-boosted trees improve on the displayed baselines and preserve some dose ordering, but an
error near 34 Gy is a substantial fraction of the 0–100 Gy range. “Best” here means best among the
tested configurations on these six studies; it does not imply metrological precision. The scatter
should be read for systematic bias, saturation, and study clusters—not only its headline score.`);
  replaceMarkdown(nb, "A predictor is more trustworthy", `
Model reliance on a compact, biologically plausible panel can improve auditability, but it does not
by itself make the predictor causal or trustworthy. The saved pipeline selected panels without
using the held-out study’s dose labels. However, the distributed 512-gene matrix was harmonized and
filtered before this notebook, so the displayed LOSO benchmark is conditional on that representation.
A fully confirmatory pipeline would repeat every supervised and data-adaptive feature-selection step
inside each training fold starting from the complete raw-count matrix.`);
  replaceMarkdown(nb, "## 12. Conclusions and honest limitations", `
## 12. Conclusions, validity, and reproducibility

**Supported findings**

- Gene expression retains moderate cross-study dose information under LOSO evaluation.
- PLS and tree ensembles are competitive on rank correlation, while parsimonious models show smaller
  within-versus-LOSO gaps in the displayed benchmark.
- Compact panels can outperform the cited seven-gene baseline within the saved analysis.
- DNA-damage annotations and stronger Zeocin than UV-C concordance support biological plausibility.

**Claim boundary**

There are only 158 samples in six studies, with one heavy-ion study and restricted dose coverage.
The effective replication count for cross-study inference is six, not 158. Absolute errors are too
large for a precision dosimeter; radiation-quality conclusions and out-of-range predictions are
exploratory. Pre-harmonization and global unsupervised filtering limit the benchmark to model
transfer on the supplied representation.

## References

[1] NASA Open Science Data Repository, “Biological Data Management Environment,” dataset records
OSD-498, OSD-502, OSD-508, OSD-510, OSD-658, and OSD-782.

[2] A. Conesa *et al.*, “A survey of best practices for RNA-seq data analysis,”
*Genome Biology*, vol. 17, 2016.

[3] D. W. Huang, B. T. Sherman, and R. A. Lempicki, “Systematic and integrative analysis of large
gene lists using DAVID bioinformatics resources,” *Nature Protocols*, vol. 4, 2009.

### Reproducibility record

All inputs needed by this notebook are included under **raddose_data/**: the harmonized matrix,
manifest, two raw-study examples, and saved interpretation tables. Run the notebook top to bottom
from this directory after installing **requirements.txt**. The notebook recomputes the core LOSO model
comparison; the more expensive panel and external-validation stages are transparently loaded from
versioned result files. No unavailable external pipeline is required to reproduce the notebook.
`);
});

console.log("Upgraded six notebooks.");
