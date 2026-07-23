import fs from "node:fs";

const file = "/Users/srikanthsamy1/Desktop/CodeProjects/research-projects/Heart Disease Cross Hospital Research.ipynb";
const nb = JSON.parse(fs.readFileSync(file, "utf8"));
const source = cell => cell.source.join("");
const setMarkdown = (prefix, text) => {
  const cell = nb.cells.find(c => c.cell_type === "markdown" && source(c).startsWith(prefix));
  if (!cell) throw new Error(`Missing markdown cell: ${prefix}`);
  cell.source = text.trim().split("\n").map((line, i, a) => i < a.length - 1 ? `${line}\n` : line);
};
const setCode = (prefix, text) => {
  const cell = nb.cells.find(c => c.cell_type === "code" && source(c).startsWith(prefix));
  if (!cell) throw new Error(`Missing code cell: ${prefix}`);
  cell.source = text.trim().split("\n").map((line, i, a) => i < a.length - 1 ? `${line}\n` : line);
  cell.execution_count = null;
  cell.outputs = [];
};

setMarkdown("**TensorFlow / Keras", `
**scikit-learn neural network.** The third model is a small multi-layer perceptron (MLP). Keeping all
three models in scikit-learn makes the notebook lighter, faster, and reproducible on an ordinary CPU.`);
setCode("import tensorflow as tf", "from sklearn.neural_network import MLPClassifier");
setMarkdown("**Last bits**", `
**Last bits** — a display helper, warning control, and folders for exported results.`);
setCode("import os, warnings", `
import os, warnings
warnings.filterwarnings("ignore")
from IPython.display import display
print("All libraries imported successfully.")`);
setCode("import random\nSEED=42", `
import random
SEED=42
random.seed(SEED); np.random.seed(SEED)
SITE_COLORS={"Cleveland":"#2c6fbb","Hungary":"#3a7d44","Switzerland":"#c0392b","VA":"#e0a13a"}
plt.rcParams.update({"figure.facecolor":"white","axes.grid":True,"grid.alpha":0.3,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "font.size":11,"axes.titlesize":12,"axes.titleweight":"bold"})
for d in ["outputs/figures","outputs/results"]: os.makedirs(d, exist_ok=True)
print("Seed set to 42 — results will be repeatable.")`);
setMarkdown("**Model 3 — a small Neural Network", `
**Model 3 — a small neural network (MLP).** Two hidden layers learn nonlinear combinations of the
measurements. L2 regularization and early stopping reduce overfitting. This remains a compact
benchmark model, not a clinical deep-learning system.`);
setCode("def build_mlp(n):", `
def build_mlp(n):
    return MLPClassifier(hidden_layer_sizes=(32,16), activation="relu", alpha=0.01,
                         learning_rate_init=1e-3, max_iter=600, early_stopping=True,
                         validation_fraction=0.2, n_iter_no_change=20, random_state=SEED)

def fit_mlp(Xtr,ytr):
    return build_mlp(Xtr.shape[1]).fit(Xtr,ytr)
print("Three models ready.")`);

for (const cell of nb.cells) {
  if (cell.cell_type !== "code") continue;
  let text = source(cell);
  text = text.replaceAll('mm.predict(Xte,verbose=0).ravel()', 'mm.predict_proba(Xte)[:,1]');
  text = text.replaceAll('fit_mlp2(Xa,ytr[a]).predict(Xb,verbose=0).ravel()', 'fit_mlp2(Xa,ytr[a]).predict_proba(Xb)[:,1]');
  text = text.replaceAll('mm.predict(Xte,verbose=0).ravel()', 'mm.predict_proba(Xte)[:,1]');
  cell.source = text.split("\n").map((line, i, a) => i < a.length - 1 ? `${line}\n` : line);
}

const travel = nb.cells.find(c => c.cell_type === "code" && source(c).startsWith("import time\ndef safe_auc"));
if (!travel) throw new Error("Missing travel evaluation cell");
let travelText = source(travel);
const start = travelText.indexOf("def fit_mlp2");
const end = travelText.indexOf("\n\nmodels=", start);
travelText = `${travelText.slice(0, start)}def fit_mlp2(X,yv):
    return build_mlp(X.shape[1]).fit(X,yv)
${travelText.slice(end + 1)}`;
travel.source = travelText.split("\n").map((line, i, a) => i < a.length - 1 ? `${line}\n` : line);
travel.execution_count = null;
travel.outputs = [];

for (const cell of nb.cells) {
  if (cell.cell_type === "code") {
    cell.execution_count = null;
    cell.outputs = [];
  }
}

fs.writeFileSync(file, `${JSON.stringify(nb, null, 1)}\n`);
console.log("Converted heart notebook to a portable scikit-learn MLP and cleared stale outputs.");
