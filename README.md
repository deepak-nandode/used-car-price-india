# Used Car Price Estimator (India)

Predicts the resale price of a used car in India from brand, age, kilometres driven, fuel, transmission, owner count and engine specs. Compares Linear Regression, Ridge and Random Forest, and serves the best model through a Streamlit web app.

**Live app:** (https://deepak-nandode-rymwmdpzcxeq5supnn3yav.streamlit.app/)  
**Data:** Kaggle, "Vehicle dataset from CarDekho" (nehalbirla), file `Car details v3.csv`

## Results (hold-out test set, 20%)
Data: 8,128 raw rows, 6,926 after removing duplicates, 6,916 used for modelling. 80/20 train/test split, seed 42.

| Model | CV R² (log) | Test R² (₹) | MAE (₹) | RMSE (₹) |
|---|---|---|---|---|
| Linear Regression | 0.882 | 0.848 | 91,856 | 178,351 |
| Ridge | 0.882 | 0.849 | 91,832 | 178,022 |
| **Random Forest** | **0.915** | **0.916** | **70,627** | **132,884** |

Random Forest wins by a clear margin; Ridge is almost identical to plain OLS because there are few features and thousands of rows, so shrinkage has little to fix. Top drivers: car age, max power, engine size, then mileage and km driven.

## Method
1. **Cleaning:** `mileage`, `engine`, `max_power` are text like "1248 CC"; the number is extracted with a regex. Exact duplicate rows are dropped *before* splitting (otherwise copies land in both train and test and inflate scores). Test-drive cars and implausible km values are removed. Brand is the first word of `name`; brands with fewer than 10 rows become "Other".
2. **Features:** age (2020 − year; the data was scraped around 2020), km driven, mileage, engine, power, seats, brand, fuel, seller type, transmission, owner.
3. **Target:** `log(1 + price)`. Prices are right-skewed and errors are proportional (₹50k off matters on a ₹2 lakh car, not on a ₹20 lakh one). Predictions are converted back to rupees before reporting.
4. **Models:** one sklearn `Pipeline` per model (median imputation, scaling, one-hot encoding, estimator), so the exact same preprocessing runs in training and in the app. Ridge picks its penalty by built-in cross-validation. The Random Forest is capped (150 trees, `min_samples_leaf=2`) to keep the file small.
5. **Selection:** the best model is chosen by 5-fold cross-validation on the training set, not by the test set. It is then refit on all data for the app.
6. **App:** the forest's individual trees give a rough 10th-90th percentile price range next to the point estimate.

## Run locally
```bash
pip install -r requirements.txt
# put "Car details v3.csv" in data/
python train.py
streamlit run app.py
```

## Deploy (Streamlit Community Cloud, free)
1. Push this folder to a public GitHub repo. **Commit `model/model.joblib`** (it is only a few MB).
2. Go to share.streamlit.io, sign in with GitHub, click "Create app", choose the repo, branch `main`, main file `app.py`, Deploy.
3. Apps sleep after a period of inactivity; the first visit afterwards is slow. Open the link before sharing it with a recruiter.

## Limitations
- Prices reflect c. 2020 listings; they are not current market values.
- Asking prices, not final sale prices.
- Rare or luxury brands have few rows, so estimates there are wider.
- Mileage mixes kmpl and km/kg (CNG/LPG); the unit is dropped.
- Possible extensions: model name as a feature, gradient boosting, log-transformed km, a SHAP explanation, conformal prediction intervals.

## Structure
```
train.py        cleaning, training, evaluation, saves model + plots
app.py          Streamlit app
model/          model.joblib, metrics.json (generated)
reports/        metrics.csv and plots (generated)
data/           put the Kaggle CSV here
```
