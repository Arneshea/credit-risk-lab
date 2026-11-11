"""Streamlit demo - enter a customer's details, get a prediction + plain-language reasons.
Run:  streamlit run app.py
"""
import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from predict import load_bundle, predict_dataframe, explain_customer
from src.data import RAW_COLS

ROOT = Path(__file__).parent
st.set_page_config(page_title="Explainable Credit Scoring", layout="wide")
model, bundle = load_bundle()
meta = bundle["meta"]

st.markdown("""
<style>
.block-container {max-width: 1120px; padding-top: 2rem;}
.hero {padding: 1.5rem 1.8rem; border-radius: 18px; background: linear-gradient(115deg,#10243a,#1c5264); color: white; margin-bottom: 1rem;}
.hero h1 {color: white; margin: 0 0 .4rem 0;}
.hero p {color: #d4e8ee; margin: 0; font-size: 1rem;}
[data-testid="stMetric"] {background: #f3f7f8; color: #17324a; padding: 1rem; border-radius: 12px;}
[data-testid="stMetric"] label, [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {color: #17324a !important;}
</style>
""", unsafe_allow_html=True)

STATUS = {"No usage that month": -2, "Paid in full": -1, "Paid minimum only (revolving)": 0,
          **{f"{m} month{'s' if m > 1 else ''} late": m for m in range(1, 9)}}
STATUS_LABELS = list(STATUS)
CODE_TO_LABEL = {v: k for k, v in STATUS.items()}


@st.cache_data
def example_customers():
    """3 real rows from the dataset: a low-, middle- and high-risk customer (according to the model)."""
    try:
        df = pd.read_csv(ROOT / "sample_customers.csv")
    except Exception:
        return {}
    names = ["Example 1 - steady payer", "Example 2 - middle", "Example 3 - often late"]
    return {n: df.iloc[i].to_dict() for i, n in enumerate(names[:len(df)])}


EX = example_customers()
st.markdown("<div class='hero'><h1>Loan risk assessment</h1><p>Explore a transparent fuzzy rule model and see the factors behind its risk indication.</p></div>", unsafe_allow_html=True)
st.info("Prototype only: this saved model was trained on Taiwan credit-card repayment data. It does not use UPI transactions or gig-worker income, and the requested loan amount does not affect its risk score. Treat the result as a demonstration, not a lending decision.")

tab1, tab2, tab3 = st.tabs(["Predict one customer", "Score a CSV of customers", "The learned rules"])

# ---------------------------------------------------------------- single customer
with tab1:
    st.subheader("Application")
    pick = st.selectbox("Start from an example (or choose 'Blank' and type your own)", ["Blank"] + list(EX))
    d = EX.get(pick, {})
    k = pick.replace(" ", "_")

    with st.form("customer_application"):
        applicant_name = st.text_input("Customer name or reference (optional)", placeholder="e.g. Application 001")
        requested_amount = st.number_input("Requested loan amount (₹)", min_value=1000, max_value=10_000_000,
                                           value=50_000, step=5000, help="Displayed with the result; the current model does not assess affordability or use this amount.")
        st.markdown("**Credit-card history used by this prototype**")
        limit = st.number_input("Credit limit in training data currency units", 1000, 5_000_000, int(d.get("LIMIT_BAL", 100_000)),
                                step=5000, key=f"{k}-limit")
        st.markdown("**Last 6 months** (month 1 = most recent)")
        hdr = st.columns([1, 3, 2, 2])
        hdr[1].caption("Payment status"); hdr[2].caption("Bill amount"); hdr[3].caption("Amount paid")
        raw = {"LIMIT_BAL": limit}
        for i in range(1, 7):
            c = st.columns([1, 3, 2, 2])
            c[0].write(f"Month {i}")
            default_label = CODE_TO_LABEL.get(int(d.get(f"PAY_{i}", 0)), "Paid minimum only (revolving)")
            s = c[1].selectbox(f"status{i}", STATUS_LABELS, index=STATUS_LABELS.index(default_label),
                               label_visibility="collapsed", key=f"{k}-s{i}")
            b = c[2].number_input(f"bill{i}", 0, 10_000_000, int(max(d.get(f"BILL_AMT{i}", 20_000), 0)),
                                  step=500, label_visibility="collapsed", key=f"{k}-b{i}")
            p = c[3].number_input(f"paid{i}", 0, 10_000_000, int(max(d.get(f"PAY_AMT{i}", 2_000), 0)),
                                  step=500, label_visibility="collapsed", key=f"{k}-p{i}")
            raw[f"PAY_{i}"], raw[f"BILL_AMT{i}"], raw[f"PAY_AMT{i}"] = STATUS[s], b, p
        st.caption("Edits are applied to the estimate when you click Predict risk.")
        submitted = st.form_submit_button("Predict risk", type="primary", use_container_width=True)

    if submitted:
        e = explain_customer(raw, model, bundle)
        st.divider()
        st.subheader("Risk indication")
        if applicant_name:
            st.caption(f"Application: {applicant_name}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Requested amount", f"₹{requested_amount:,.0f}")
        m2.metric("Model risk estimate", f"{e['risk']:.0%}")
        m3.metric("Model indication", e["decision"])
        if e["decision"] == "Approve":
            st.success("Lower-risk indication: below this model's validation threshold. This is not a loan approval.")
        else:
            st.warning("Higher-risk indication: refer for human review. This is not an automatic rejection.")

        st.subheader("Why this prediction")
        if e["reasons"]:
            for r in e["reasons"]:
                st.write(("🔺 " if r["supports_default"] else "✅ ") + r["text"])
                st.caption(f"{r['rule']}   (rule strength {r['strength']:.2f})")
        else:
            st.write("No strong rule fired, so the score stays close to the population average.")
        st.subheader("What would help most")
        if e["tips"]:
            for t in e["tips"]:
                st.write("💡 " + t["text"])
        else:
            st.write("Nothing obvious to improve - the main drivers are already favourable.")

# ---------------------------------------------------------------- batch
with tab2:
    st.write("Upload a CSV with one customer per row. Required columns: " + ", ".join(RAW_COLS))
    st.download_button("Download blank template", (ROOT / "customer_template.csv").read_bytes(), "customer_template.csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            raw_df = pd.read_csv(up)
            res = predict_dataframe(raw_df, model, bundle)
            out = pd.concat([raw_df.reset_index(drop=True), res[["risk", "credit_score", "decision"]]], axis=1)
            st.dataframe(out[["risk", "credit_score", "decision"] + RAW_COLS], use_container_width=True)
            st.download_button("Download scored CSV", out.to_csv(index=False), "scored_customers.csv")
        except Exception as ex:
            st.error(f"Could not score this file: {ex}")

# ---------------------------------------------------------------- rules
with tab3:
    st.write(f"The whole model is these {model.n_rules()} rules (decision threshold on risk: {bundle['threshold']:.2f}).")
    for r in np.where(model.active_mask())[0]:
        st.write(f"**R{r + 1}.** " + model.rule_text(r, meta["labels"]))
    img = ROOT / "results" / "membership_functions_taiwan.png"
    if img.exists():
        st.image(str(img), caption="Evolved meaning of Low / Medium / High for each feature")
