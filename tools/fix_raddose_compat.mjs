import fs from "node:fs";

const file = "/Users/srikanthsamy1/Desktop/CodeProjects/research-projects/raddose-phytodosimetry/RadDose_Full_Walkthrough.ipynb";
const nb = JSON.parse(fs.readFileSync(file, "utf8"));
let changed = 0;
for (const cell of nb.cells) {
  if (cell.cell_type !== "code") continue;
  const original = cell.source.join("");
  const updated = original
    .replace("LassoCV(n_alphas=20, max_iter=5000, random_state=0)",
             "LassoCV(alphas=np.logspace(-4, 1, 20), max_iter=5000, random_state=0)")
    .replace("ElasticNetCV(n_alphas=20, max_iter=5000, random_state=0)",
             "ElasticNetCV(alphas=np.logspace(-4, 1, 20), max_iter=5000, random_state=0)");
  if (updated !== original) {
    cell.source = updated.split("\n").map((line, i, all) => i < all.length - 1 ? `${line}\n` : line);
    cell.execution_count = null;
    cell.outputs = [];
    changed += 1;
  }
}
if (changed > 1) throw new Error(`Expected at most one model-roster cell, found ${changed}`);

const legacyConclusion = nb.cells.find(
  cell => cell.cell_type === "markdown" && cell.source.join("").startsWith("**What the project shows**")
);
if (!legacyConclusion) throw new Error("Missing legacy conclusion cell");
const exercises = `
### Student extensions

1. Recompute the LOSO table one held-out study at a time. Which study dominates each model’s error,
   and what design difference might explain it?
2. Replace the random within-study split with group-aware folds and explain why the optimistic
   ceiling changes.
3. Plot residuals against dose and radiation quality. Look for saturation, heteroscedasticity, and
   systematic underprediction at high doses.
4. Repeat the model ranking with MAE rather than Spearman correlation. Why can the two leaderboards
   disagree?
5. Design a fully nested pipeline that begins from all genes and performs feature filtering using
   training studies only. Identify every quantity that must be fitted inside the fold.
`.trim();
legacyConclusion.source = exercises.split("\n").map(
  (line, i, all) => i < all.length - 1 ? `${line}\n` : line
);
fs.writeFileSync(file, `${JSON.stringify(nb, null, 1)}\n`);
console.log("Made the RadDose regularization grid version-stable.");
