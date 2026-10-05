import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Used Car Price Estimator (India)", page_icon="🚗")


@st.cache_resource
def load():
    bundle = joblib.load(Path("model/model.joblib"))
    metrics = json.load(open("model/metrics.json"))
    return bundle, metrics


bundle, metrics = load()
pipe, opt = bundle["pipeline"], bundle["options"]
d = opt["defaults"]

st.title("🚗 Used Car Price Estimator (India)")
st.caption("Trained on ~8k CarDekho listings (c. 2020). Estimates are in 2020-era rupees.")

c1, c2 = st.columns(2)
brand = c1.selectbox("Brand", opt["brands"], index=opt["brands"].index("Maruti") if "Maruti" in opt["brands"] else 0)
age = c2.slider("Car age (years)", 0, 25, 5)
km = c1.number_input("Kilometres driven", 0, 500_000, 60_000, step=5_000)
fuel = c2.selectbox("Fuel", opt["fuel"])
seller = c1.selectbox("Seller type", opt["seller_type"])
trans = c2.selectbox("Transmission", opt["transmission"])
owner = c1.selectbox("Owner", opt["owner"])
seats = c2.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10, 14], index=2)
engine = c1.number_input("Engine (CC)", 600, 5000, int(d["engine"]), step=50)
power = c2.number_input("Max power (bhp)", 30, 500, int(d["max_power"]), step=5)
mileage = st.number_input("Mileage (kmpl)", 5.0, 40.0, float(round(d["mileage"], 1)), step=0.5)

if st.button("Estimate price", type="primary"):
    row = pd.DataFrame([{
        "age": age, "km_driven": km, "mileage": mileage, "engine": engine,
        "max_power": power, "seats": seats, "brand": brand, "fuel": fuel,
        "seller_type": seller, "transmission": trans, "owner": owner}])
    point = float(np.expm1(pipe.predict(row))[0])
    st.metric("Estimated price", f"₹ {point:,.0f}")

    model = pipe.named_steps["model"]
    if hasattr(model, "estimators_"):   # random forest: spread across trees = rough range
        Z = pipe.named_steps["prep"].transform(row)
        per_tree = np.expm1([t.predict(Z)[0] for t in model.estimators_])
        lo, hi = np.percentile(per_tree, [10, 90])
        st.write(f"Likely range (10th–90th percentile of trees): ₹ {lo:,.0f} – ₹ {hi:,.0f}")

with st.expander("How good is this model?"):
    st.write(f"Model used: **{metrics['best']}** · rows: {metrics['n_rows']:,}")
    st.dataframe(pd.DataFrame(metrics["table"]).set_index("model"))
    st.caption("Metrics are on a 20% hold-out test set. MAE/RMSE are in rupees.")
