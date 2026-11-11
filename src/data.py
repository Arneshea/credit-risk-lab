"""Dataset loading + 'alternative-data style' feature engineering.

Every engineered feature is scaled to [0, 1] so that one shared fuzzy partition
(Low / Medium / High) can be used for all of them.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---- metadata used for explanations (labels, good direction, how to show raw units) ----
META = {
    "taiwan": {
        "features": ["late_severity", "recent_delay", "repayment_ratio",
                     "utilization", "activity", "cashflow_regularity"],
        "labels": {
            "late_severity": "average payment delay (last 6 months)",
            "recent_delay": "delay in the most recent month",
            "repayment_ratio": "share of billed amount actually repaid",
            "utilization": "share of credit limit in use",
            "activity": "months of active usage",
            "cashflow_regularity": "regularity of repayment amounts",
        },
        # which way is 'good' for the applicant (None = not something to advise on)
        "direction": {"late_severity": "lower", "recent_delay": "lower",
                      "repayment_ratio": "higher", "utilization": "lower",
                      "activity": None, "cashflow_regularity": "higher"},
        # (multiplier, template) to show the normalised value in human units
        "unit": {"late_severity": (3, "{:.1f} months"), "recent_delay": (3, "{:.0f} months late"),
                 "repayment_ratio": (100, "{:.0f}%"), "utilization": (100, "{:.0f}%"),
                 "activity": (6, "{:.0f} of 6 months"), "cashflow_regularity": (1, "{:.2f} (0-1)")},
    },
    "german": {
        "features": ["account_risk", "history_risk", "savings_risk",
                     "employment_length", "loan_duration", "installment_burden"],
        "labels": {
            "account_risk": "checking-account status risk",
            "history_risk": "past credit history risk",
            "savings_risk": "savings level risk",
            "employment_length": "length of employment",
            "loan_duration": "loan duration",
            "installment_burden": "installment burden",
        },
        "direction": {"account_risk": "lower", "history_risk": "lower", "savings_risk": "lower",
                      "employment_length": "higher", "loan_duration": "lower",
                      "installment_burden": "lower"},
        "unit": {"account_risk": (1, "{:.2f}"), "history_risk": (1, "{:.2f}"),
                 "savings_risk": (1, "{:.2f}"), "employment_length": (1, "{:.2f}"),
                 "loan_duration": (72, "{:.0f} months"), "installment_burden": (1, "{:.2f}")},
    },
}


RAW_COLS = (["LIMIT_BAL"] + [f"PAY_{i}" for i in range(1, 7)]
            + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)])


def taiwan_features(df):
    """Raw customer details (credit limit + 6 months of status / bill / payment) -> 6 fuzzy-ready features.
    Used for training AND for predicting new customers, so both always go through the same code."""
    pay = df[[f"PAY_{i}" for i in range(1, 7)]].to_numpy(float)
    bill = df[[f"BILL_AMT{i}" for i in range(1, 7)]].to_numpy(float)
    paid = df[[f"PAY_AMT{i}" for i in range(1, 7)]].to_numpy(float)
    limit = np.maximum(df["LIMIT_BAL"].to_numpy(float), 1.0)

    late = np.clip(pay, 0, None)                    # months of delay (<=0 means on time)
    bill_pos = np.clip(bill, 0, None)
    sum_bill = bill_pos.sum(1)
    mean_paid, std_paid = paid.mean(1), paid.std(1)
    cv = np.where(mean_paid > 0, std_paid / np.maximum(mean_paid, 1e-9), 1.0)

    return pd.DataFrame({
        "late_severity": np.clip(late.mean(1) / 3, 0, 1),
        "recent_delay": np.clip(late[:, 0] / 3, 0, 1),
        "repayment_ratio": np.clip(np.where(sum_bill > 0, paid.sum(1) / np.maximum(sum_bill, 1), 1.0), 0, 1),
        "utilization": np.clip(bill_pos.mean(1) / limit, 0, 1),
        "activity": (bill > 0).mean(1),
        "cashflow_regularity": 1 - np.clip(cv / 2, 0, 1),   # proxy for income regularity
    })


def load_taiwan():
    df = pd.read_csv(DATA_DIR / "taiwan_credit.csv")
    return taiwan_features(df), df["default"].to_numpy(int)


GERMAN_COLS = ["status", "duration", "credit_history", "purpose", "amount", "savings",
               "employment", "installment_rate", "personal_status", "other_debtors",
               "residence_since", "property", "age", "other_installment", "housing",
               "existing_credits", "job", "liable_people", "telephone", "foreign_worker", "cls"]


def load_german_raw():
    df = pd.read_csv(DATA_DIR / "german_credit.csv", header=None, names=GERMAN_COLS)
    return df, (df["cls"] == 2).to_numpy(int)          # 2 = bad credit risk


def _target_encode(train_col, y_train, other_col, smoothing=5):
    prior = y_train.mean()
    st = pd.DataFrame({"c": train_col.values, "y": y_train}).groupby("c")["y"].agg(["mean", "count"])
    smooth = (st["count"] * st["mean"] + smoothing * prior) / (st["count"] + smoothing)
    return other_col.map(smooth).fillna(prior).to_numpy(float), smooth


def german_features(df_train, y_train, df_other):
    """Encode using training rows only (no leakage into val/test)."""
    out = {}
    for name, col in [("account_risk", "status"), ("history_risk", "credit_history"),
                      ("savings_risk", "savings")]:
        tr, smooth = _target_encode(df_train[col], y_train, df_train[col])
        lo, hi = smooth.min(), smooth.max()
        ot, _ = _target_encode(df_train[col], y_train, df_other[col])
        out[name] = (np.clip((ot - lo) / (hi - lo + 1e-9), 0, 1))
    emp = df_other["employment"].map({"A71": 0, "A72": .25, "A73": .5, "A74": .75, "A75": 1.0})
    out["employment_length"] = emp.to_numpy(float)
    out["loan_duration"] = np.clip(df_other["duration"].to_numpy(float) / 72, 0, 1)
    out["installment_burden"] = (df_other["installment_rate"].to_numpy(float) - 1) / 3
    return pd.DataFrame(out)[META["german"]["features"]]


def prepare(name, seed=0):
    """Stratified 60/20/20 train/val/test split. Returns dict of arrays."""
    if name == "taiwan":
        X, y = load_taiwan()
        idx = np.arange(len(y))
        i_tr, i_tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
        i_va, i_te = train_test_split(i_tmp, test_size=0.5, stratify=y[i_tmp], random_state=seed)
        Xtr, Xva, Xte = X.iloc[i_tr], X.iloc[i_va], X.iloc[i_te]
    elif name == "german":
        df, y = load_german_raw()
        idx = np.arange(len(y))
        i_tr, i_tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
        i_va, i_te = train_test_split(i_tmp, test_size=0.5, stratify=y[i_tmp], random_state=seed)
        Xtr = german_features(df.iloc[i_tr], y[i_tr], df.iloc[i_tr])
        Xva = german_features(df.iloc[i_tr], y[i_tr], df.iloc[i_va])
        Xte = german_features(df.iloc[i_tr], y[i_tr], df.iloc[i_te])
    else:
        raise ValueError(name)
    return dict(Xtr=Xtr.to_numpy(float), ytr=y[i_tr], Xva=Xva.to_numpy(float), yva=y[i_va],
                Xte=Xte.to_numpy(float), yte=y[i_te], features=list(Xtr.columns))
