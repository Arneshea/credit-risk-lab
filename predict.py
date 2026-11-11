"""Prediction for NEW customers (raw details in -> score, decision, reasons, tips out).

Command line:
    python predict.py --csv sample_customers.csv            # batch: one row per customer
    python predict.py --csv sample_customers.csv --out scored.csv
    python predict.py --demo                                 # score 3 example customers
Input columns (same as the Taiwan dataset):
    LIMIT_BAL, PAY_1..PAY_6, BILL_AMT1..BILL_AMT6, PAY_AMT1..PAY_AMT6
    month 1 = most recent. PAY_x: -2 no usage, -1 paid in full, 0 paid minimum/revolving, 1..9 = months late
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import RAW_COLS, taiwan_features
from src.fuzzy import FuzzyRuleBase
from src.explain import explain, format_text

ROOT = Path(__file__).resolve().parent


def load_bundle(path=None):
    b = json.loads((Path(path) if path else ROOT / "results" / "model_taiwan.json").read_text())
    return FuzzyRuleBase.from_dict(b["model"]), b


def predict_dataframe(raw, model=None, bundle=None):
    """raw: DataFrame with RAW_COLS. Returns DataFrame with features, risk, credit_score, decision."""
    if model is None:
        model, bundle = load_bundle()
    missing = [c for c in RAW_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    feats = taiwan_features(raw)
    risk = model.score(feats.to_numpy())
    out = feats.copy()
    out["risk"] = risk.round(3)
    out["credit_score"] = np.rint(100 * (1 - risk)).astype(int)
    out["decision"] = np.where(risk < bundle["threshold"], "Approve", "Refer for review")
    return out


def explain_customer(raw_row, model=None, bundle=None):
    """raw_row: dict or 1-row DataFrame of RAW_COLS. Returns the explanation dict from src.explain."""
    if model is None:
        model, bundle = load_bundle()
    df = pd.DataFrame([raw_row]) if isinstance(raw_row, dict) else raw_row.iloc[[0]]
    x = taiwan_features(df).to_numpy()[0]
    return explain(model, x, bundle["meta"], bundle["threshold"], bundle["good_values"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV of customers (columns listed in the module docstring)")
    ap.add_argument("--out", help="write scored CSV here")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    model, bundle = load_bundle()
    if a.demo or not a.csv:
        raw = pd.read_csv(ROOT / "sample_customers.csv")
    else:
        raw = pd.read_csv(a.csv)
    scored = predict_dataframe(raw, model, bundle)
    print(scored[["risk", "credit_score", "decision"]].to_string())
    if a.out:
        pd.concat([raw, scored], axis=1).to_csv(a.out, index=False)
        print(f"saved -> {a.out}")
    if len(raw) <= 5:
        for i in range(len(raw)):
            print(f"\n--- customer {i + 1} ---")
            print(format_text(explain_customer(raw.iloc[[i]], model, bundle)))
