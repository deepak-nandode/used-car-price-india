"""Train and compare Linear Regression, Ridge and Random Forest on used-car prices.

Run:  python train.py
Needs: data/Car details v3.csv  (Kaggle: nehalbirla/vehicle-dataset-from-cardekho)
Writes: model/model.joblib, model/metrics.json, reports/*.png, reports/metrics.csv
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("data/Car details v3.csv")
REF_YEAR = 2020            # dataset was scraped around 2020, so age = 2020 - year
SEED = 42
NUM = ["age", "km_driven", "mileage", "engine", "max_power", "seats"]
CAT = ["brand", "fuel", "seller_type", "transmission", "owner"]


def to_number(s: pd.Series) -> pd.Series:
    """'1248 CC' -> 1248.0 ; '23.4 kmpl' -> 23.4 ; ' bhp' (blank) -> NaN"""
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce")


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Raw shape: {df.shape}")
    print("Columns:", list(df.columns))

    df = df.drop_duplicates()                     # the file has many exact duplicates;
    print(f"After dropping duplicates: {df.shape}")  # keeping them leaks rows into the test set

    df["mileage"] = to_number(df["mileage"])
    df["engine"] = to_number(df["engine"])
    df["max_power"] = to_number(df["max_power"])
    df["age"] = REF_YEAR - df["year"]

    first = df["name"].str.split().str[0]
    df["brand"] = first.where(first != "Land", "Land Rover")

    df = df[(df["owner"] != "Test Drive Car") & (df["km_driven"] < 500_000) & (df["age"] >= 0)]
    counts = df["brand"].value_counts()
    rare = counts[counts < 10].index
    df["brand"] = df["brand"].replace(dict.fromkeys(rare, "Other"))
    print(f"Final modelling rows: {len(df)}")
    return df


def make_pipeline(model) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("prep", pre), ("model", model)])


def evaluate(name, pipe, X_tr, y_tr_log, X_te, y_te_log) -> dict:
    cv = cross_val_score(pipe, X_tr, y_tr_log, cv=5, scoring="r2", n_jobs=-1)
    pipe.fit(X_tr, y_tr_log)
    pred_log = pipe.predict(X_te)
    y_true, y_pred = np.expm1(y_te_log), np.expm1(pred_log)   # back to rupees
    return {
        "model": name,
        "cv_r2_log": round(cv.mean(), 4),
        "test_r2_log": round(r2_score(y_te_log, pred_log), 4),
        "test_r2_rupees": round(r2_score(y_true, y_pred), 4),
        "test_mae_rupees": round(mean_absolute_error(y_true, y_pred)),
        "test_rmse_rupees": round(float(np.sqrt(mean_squared_error(y_true, y_pred)))),
    }


def main():
    if not DATA.exists():
        sys.exit(f"Put the Kaggle CSV at {DATA}")
    df = load_and_clean(DATA)
    X, y = df[NUM + CAT], np.log1p(df["selling_price"])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    pipes = {
        "Linear Regression": make_pipeline(LinearRegression()),
        "Ridge": make_pipeline(RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "Random Forest": make_pipeline(RandomForestRegressor(
            n_estimators=150, min_samples_leaf=2, max_features=0.5,
            n_jobs=-1, random_state=SEED)),
    }
    rows = [evaluate(n, p, X_tr, y_tr, X_te, y_te) for n, p in pipes.items()]
    res = pd.DataFrame(rows).set_index("model")
    print("\n", res.to_string())

    best = res["cv_r2_log"].idxmax()     # choose by cross-validation, NOT by the test set
    print(f"\nBest by CV: {best}")
    best_pipe = pipes[best]

    # Actual vs predicted plot (test set)
    pred = np.expm1(best_pipe.predict(X_te)); true = np.expm1(y_te)
    plt.figure(figsize=(5, 5))
    plt.scatter(true / 1e5, pred / 1e5, s=6, alpha=0.4)
    lim = [0, np.percentile(true, 99) / 1e5]
    plt.plot(lim, lim, "r--"); plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("Actual price (lakh ₹)"); plt.ylabel("Predicted price (lakh ₹)")
    plt.title(f"{best}: actual vs predicted (test)")
    plt.tight_layout(); plt.savefig("reports/actual_vs_predicted.png", dpi=150); plt.close()

    # Top drivers (random forest only)
    rf = pipes["Random Forest"]
    names = rf.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).nlargest(10)
    imp.iloc[::-1].plot.barh(figsize=(6, 4), title="Random Forest: top 10 features")
    plt.tight_layout(); plt.savefig("reports/feature_importance.png", dpi=150); plt.close()

    # Refit best model on ALL data for the app
    best_pipe.fit(X, y)
    options = {
        "brands": sorted(df["brand"].unique()),
        "fuel": sorted(df["fuel"].unique()),
        "seller_type": sorted(df["seller_type"].unique()),
        "transmission": sorted(df["transmission"].unique()),
        "owner": sorted(df["owner"].unique()),
        "defaults": {c: float(df[c].median()) for c in ["mileage", "engine", "max_power", "seats"]},
    }
    joblib.dump({"pipeline": best_pipe, "options": options, "best": best},
                "model/model.joblib", compress=3)
    res.to_csv("reports/metrics.csv")
    json.dump({"best": best, "table": res.reset_index().to_dict("records"),
               "n_rows": len(df)}, open("model/metrics.json", "w"), indent=2)
    size = Path("model/model.joblib").stat().st_size / 1e6
    print(f"Saved model/model.joblib ({size:.1f} MB)")


if __name__ == "__main__":
    main()
