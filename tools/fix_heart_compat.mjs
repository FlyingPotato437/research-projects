import fs from "node:fs";

const file = "/Users/srikanthsamy1/Desktop/CodeProjects/research-projects/Heart Disease Cross Hospital Research.ipynb";
const nb = JSON.parse(fs.readFileSync(file, "utf8"));
let replacements = 0;
for (const cell of nb.cells) {
  if (cell.cell_type !== "code") continue;
  const original = cell.source.join("");
  let updated = original.replace('boxplot(g,labels=', 'boxplot(g,tick_labels=');
  updated = updated.replace('Part 2 — travel test (10 shared features):',
                            'Part 2 — travel test (9 portable features):');
  updated = updated.replace('ax.set_title("Robustness to a new hospital — smaller gap = more robust")',
                            'ax.set_title("Internal vs external discrimination (macro averages)")');
  updated = updated.replace(
    'print("Most robust to a new hospital (smallest gap):", gap_df.loc[gap_df["gap"].idxmin(),"model"])\\nprint("Least robust (biggest gap):", gap_df.loc[gap_df["gap"].idxmax(),"model"])',
    'print("Highest mean new-hospital AUC:", gap_df.loc[gap_df["new_hospital_AUC"].idxmax(),"model"])\\nprint("Largest positive internal-minus-external difference:", gap_df.loc[gap_df["gap"].idxmax(),"model"])'
  );
  if (updated !== original) {
    cell.source = updated.split("\n").map((line, i, all) => i < all.length - 1 ? `${line}\n` : line);
    cell.execution_count = null;
    cell.outputs = [];
    replacements += 1;
  }
}

const setMarkdown = (prefix, text) => {
  const cell = nb.cells.find(c => c.cell_type === "markdown" && c.source.join("").startsWith(prefix));
  if (!cell) throw new Error(`Missing markdown cell: ${prefix}`);
  cell.source = text.trim().split("\n").map((line, i, all) => i < all.length - 1 ? `${line}\n` : line);
};
setMarkdown("For each model we average the **same-hospital**", `
For each model we report the macro-average internal AUC and the macro-average over directed
new-hospital transfers. Their signed difference is **internal minus external AUC**: positive means
the external average is lower; negative means it is higher. This is a descriptive contrast, not a
pure robustness statistic, because the internal and external test sets differ in both hospital and
difficulty. Read it together with the full transfer matrix and the absolute external AUC.`);
setMarkdown("### What we found", `
### What the corrected experiment supports

- **Internal Cleveland discrimination is strong on this benchmark.** The best five-fold result is
  approximately 0.91 ROC-AUC using all 13 Cleveland variables.
- **Transfer is heterogeneous rather than uniformly worse.** Some directed hospital pairs lose
  discrimination and others improve; a single average hides this asymmetry.
- **Random forest has the highest macro-average external AUC in this run.** Logistic regression has
  the largest positive internal-minus-external difference, while the MLP has lower absolute
  performance despite a small signed difference.
- **Source cohort quality matters.** Small size, severe class imbalance, and site-specific missingness
  make Switzerland and VA especially uncertain training sources.

### Limitations

- Part I uses 13 Cleveland variables; Part II uses nine portable variables, so their scores answer
  different questions.
- Four historical cohorts do not establish clinical generalization. There are no confidence
  intervals, calibration curves, threshold-specific operating points, subgroup audits, or prospective
  validation.
- Macro-averaging gives every directed hospital pair equal weight and does not account for correlated
  pairs or different target sizes.

> **Takeaway:** external validation is a matrix of source–target behaviors, not a badge earned from one
> held-out split. Model selection should consider absolute target-site performance, uncertainty, and
> clinical utility—not only the internal-minus-external difference.`);
fs.writeFileSync(file, `${JSON.stringify(nb, null, 1)}\n`);
console.log(`Updated heart compatibility and interpretation (${replacements} code cells).`);
