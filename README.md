# Explainable Fuzzy-GA Credit Scoring for Thin-File Applicants

Soft Computing PBL (26B16CS311, CO4) - modification of the genetic fuzzy rule learner of
Hoffmann, Baesens, Mues, Van Gestel & Vanthienen (EJOR 177(1), 2007).

**Idea:** combine fuzzy inference, evolutionary search, and neural learning in a neuro-fuzzy credit
default classifier. A genetic algorithm (GA) searches fuzzy rule antecedents and Gaussian membership
sets. Backpropagation then fine-tunes the membership centers, widths, and zero-order Sugeno rule-output
logits. A multilayer perceptron (MLP) is trained on the same split as a neural baseline.

The GA fitness rewards held-out-from-search discrimination while penalizing complex rule bases:
`fitness = training-subsample AUC - lambda * normalized rule complexity`. Its best rule set initializes
the neuro-fuzzy system; neural backpropagation updates its premise and consequent parameters by binary
cross-entropy. The web comparison reports a seed-0 held-out test result, not a multi-seed estimate.

## Run the custom frontend
```bash
pip install -r requirements.txt
python download_data.py          # only needed if the bundled data/ files are missing
python web_app.py
# open http://127.0.0.1:8765
```

The custom UI has two switchable modes: **Model comparison** trains the GA-optimized neuro-fuzzy
classifier and an ANN baseline on the same training split; **Legacy score demo** keeps the original
saved Taiwan fuzzy-GA scoring flow available. Each model gets a threshold selected on validation data;
reported metrics use the held-out test split. Results are cached for the life of the local server per
dataset.

The original Streamlit app remains available separately:
```bash
streamlit run app.py
```
The full multi-seed experiment suite can still be run with `python run_experiments.py` (about 10 min).

## Predicting for a new customer
* **App:** `streamlit run app.py` -> tab *Predict one customer*: enter the credit limit and, for the last 6 months,
  payment status, bill amount and amount paid. You get the credit score, default risk, decision, the fuzzy rules
  that fired and "what would help most" tips. Tab *Score a CSV* does the same for many customers
  (template: `customer_template.csv`).
* **Command line:** `python predict.py --csv sample_customers.csv --out scored.csv`
* **From code:** `from predict import explain_customer; explain_customer({...raw fields...})`
* Month 1 = most recent. Status codes: -2 no usage, -1 paid in full, 0 paid minimum/revolving, 1-8 = months late.

## Layout
| Path | What |
|---|---|
| `src/data.py` | loading, feature engineering, stratified 60/20/20 split |
| `src/fuzzy.py` | Low/Medium/High partitions, rule base, inference (AND = min) |
| `src/ga.py` | genetic learner (tournament selection, crossover, mutation, elitism, pruning) |
| `src/neuro_fuzzy.py` | GA search over fuzzy antecedents, then backpropagation over Gaussian memberships and Sugeno outputs |
| `src/explain.py` | rule-based reasons + "what would help" what-if tips |
| `src/lab.py` | reproducible held-out comparison of GA neuro-fuzzy and ANN models |
| `run_experiments.py` | all experiments, baselines, plots |
| `web_app.py` and `frontend/` | custom HTML/CSS/JS interface with model-comparison and legacy modes |
| `predict.py` | scoring + explanation for new customers (used by the app and the CLI) |
| `app.py` | Streamlit demo (single customer form, CSV batch scoring, rule viewer) |
| `results/` | `summary.md`, `metrics_raw.csv`, ROC / convergence / membership plots, `model_*.json`, `sample_explanations_*.txt` |

## Data
* **Taiwan "Default of Credit Card Clients"** (UCI; 24,000-row public copy, 22% default). Main dataset.
  Six behaviour features are derived from 6 months of history: average delay, most recent delay,
  share of bills repaid, credit utilisation, months of active use, regularity of repayment amounts
  (used as an income-regularity proxy).
* **German Credit (UCI Statlog)**: 1,000 rows, 30% bad. Benchmark used in the base paper's line of work.
  Categorical fields are target-encoded using training rows only.
* Neither dataset contains gig workers or UPI histories. The project is a soft-computing method
  demonstration motivated by thin-file lending; it does not establish performance for gig workers.
* Home Credit Default Risk (Kaggle) needs a Kaggle login, so it is not bundled. To add it, write a loader
  in `src/data.py` that returns six [0,1] features (e.g. aggregated installment lateness) and add a `META` entry.

## Results (held-out test sets, mean +/- std over seeds; threshold picked on validation set)
| Dataset | Model | AUC | Balanced acc. | Recall (defaulters) | #Rules | Conds/rule |
|---|---|---|---|---|---|---|
| Taiwan (5 splits) | Logistic Regression | 0.752 +/- 0.010 | 0.702 | 0.607 | - | - |
| | Random Forest (black box) | 0.751 +/- 0.005 | 0.697 | 0.586 | - | - |
| | GA-Fuzzy base (accuracy-only) | 0.757 +/- 0.010 | 0.703 | 0.595 | 5.6 | 3.02 |
| | **GA-Fuzzy proposed** | 0.756 +/- 0.009 | 0.705 | 0.622 | **3.0** | **1.75** |
| German (10 splits) | Logistic Regression | 0.775 +/- 0.033 | 0.700 | 0.715 | - | - |
| | Random Forest (black box) | 0.764 +/- 0.044 | 0.695 | 0.657 | - | - |
| | GA-Fuzzy base (accuracy-only) | 0.764 +/- 0.044 | 0.698 | 0.750 | 6.3 | 2.76 |
| | **GA-Fuzzy proposed** | 0.763 +/- 0.053 | 0.689 | 0.735 | **3.6** | **2.04** |

**Honest reading:** accuracy is on par with the black-box baselines; the gain is *compactness*
(about half the rules and fewer conditions per rule for the same AUC). On German the +/- is large
(only 200 test rows per split), so differences in accuracy between models there are within noise.

The table above documents the original saved fuzzy-GA experiment, which remains available in the
legacy mode. The custom interface is the integrated GA neuro-fuzzy project: it trains that model and
an ANN on the same seed-0 split and reports the results live. Do not compare that single split with
the legacy table's mean +/- standard deviation as if they were the same experiment.

## Known limitations (say these before the panel asks)
* Features are proxies (see Data). Accuracy on real gig-worker data is untested.
* The final risk is the share of firing strength that supports "default", so a very weakly fired rule can
  still produce an extreme score (see Applicant B in `sample_explanations_taiwan.txt`). A calibration step
  (e.g. isotonic regression on the validation set) is a natural future-scope item.
* The GA is trained on a stratified subsample of up to 5,000 training rows for speed.
* Results were produced with the default hyper-parameters (pop 60, 60 generations, lambda 0.05); lambda was
  not tuned, so a lambda sweep would be a good extra experiment for the report.
