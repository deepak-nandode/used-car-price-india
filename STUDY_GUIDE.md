# 12-day study plan (about 1.5-2 h/day)
You already have the stats base (regression, residuals, bias-variance). This plan maps it onto the code.

| Day | Do | Goal |
|---|---|---|
| 1 | Download the CSV. In a notebook run `df.shape`, `df.info()`, `df.describe()`, `df.head()` | Know the columns and the 8k-row size |
| 2 | Read `to_number` and `load_and_clean` in train.py; reproduce each step by hand on a few rows | Explain every cleaning decision |
| 3 | Plot price vs log price histograms | Justify the log target in one sentence |
| 4 | Fit plain LinearRegression yourself on 3 numeric columns, no pipeline | Read coefficients, relate to OLS from class |
| 5 | Read about one-hot encoding, scaling, imputation; read `make_pipeline` | Why a Pipeline avoids leakage |
| 6 | Ridge: vary alpha manually and watch coefficients and CV score | Explain the penalty in terms of bias-variance |
| 7 | Random forest: bagging, `min_samples_leaf`, feature importance | Explain why it beats linear here (interactions, non-linearity) |
| 8 | Metrics: compute R², MAE, RMSE by hand in numpy | Say when MAE vs RMSE matters |
| 9 | Run `python train.py`; fill the README table; look at the plots | Interpret residuals and the actual-vs-predicted plot |
| 10 | Run the Streamlit app locally; read `app.py`; change a widget | Understand the app loop (script reruns on each interaction) |
| 11 | Make a GitHub repo, push, deploy on Streamlit Cloud | Live link |
| 12 | Rehearse the interview answers below | 2-minute project pitch |

## Likely interview questions
- **Why log of price?** Right-skewed target, multiplicative errors, stabilises variance, keeps predictions positive.
- **Why drop duplicates before the split?** Identical rows in train and test leak answers and inflate the score.
- **Why choose the model by CV, not by test score?** Test set must stay untouched to give an honest final estimate.
- **Why did the random forest win (or not)?** Price depends on interactions (brand x age x km) and non-linear depreciation; linear models need hand-built terms.
- **Ridge vs OLS?** Shrinks correlated coefficients (engine, power, price tiers), lowers variance; here the gain is likely small because there are few features and many rows. Say what your results show.
- **How would you improve it?** Model-level features, gradient boosting, time-aware validation, interval calibration.
- **What are the limits?** 2020 asking prices, selection bias in listings, rare brands.

## Resume bullets (edit numbers after you run it)
- Built and deployed an end-to-end used-car price predictor in India (8,128 CarDekho listings, 6.9k after de-duplication) with a Streamlit web app on Streamlit Community Cloud.
- Cleaned unit-embedded text fields, removed duplicate-row leakage, modelled log-price, and compared Linear, Ridge and Random Forest in leakage-safe sklearn pipelines; Random Forest reached R² = 0.92, MAE ≈ ₹70.6k, RMSE ≈ ₹133k on a held-out test set.
- Added a prediction range from the forest's tree spread and reported R², MAE and RMSE in rupees.
