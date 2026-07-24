"""Trains the minprice predictor and saves it to model.pkl.

Re-implements the bagging/random-forest model from pricepredictor.R
(MSE ~215-320 depending on split) using scikit-learn. RandomForestRegressor
is used (rather than GradientBoostingRegressor, which scored slightly
better) because it's one of the model types m2cgen can export to
dependency-free pure Python -- see export_model.py -- which keeps the
deployed API's bundle small enough for Vercel's serverless size limit.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

NUMERIC_FEATURES = ["weekend", "pop", "score", "month"]
CATEGORICAL_FEATURES = ["genre"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "minprice"

df = pd.read_csv("mintry.csv")

X = df[FEATURES]
y = df[TARGET]

preprocessor = ColumnTransformer(
    transformers=[
        ("genre", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ],
    remainder="passthrough",
)


def build_model():
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=100, max_features=3, random_state=12345
                ),
            ),
        ]
    )


# A single train/test split is noisy on a dataset this small (923 rows), so use
# 10-fold CV to get a stable estimate of how well the model generalizes.
kf = KFold(n_splits=10, shuffle=True, random_state=12345)
scores = cross_val_score(build_model(), X, y, cv=kf, scoring="neg_mean_squared_error")
print(f"10-fold CV MSE: {-scores.mean():.4f} (+/- {scores.std():.4f})")

# Fit the model actually served by the API on all available data.
model = build_model()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Saved model.pkl")
