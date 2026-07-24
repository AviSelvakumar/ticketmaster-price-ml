"""Exports the trained gbm regressor to dependency-free pure Python code
(via m2cgen) so the prediction API can run without numpy/pandas/scikit-learn
at request time -- needed to fit Vercel's serverless function size limit.

Run after train_model.py (which produces model.pkl).
"""
import sys

import joblib
import m2cgen

sys.setrecursionlimit(1_000_000)

model = joblib.load("model.pkl")
rf = model.named_steps["rf"]

code = m2cgen.export_to_python(rf)

with open("api/ml_model.py", "w") as f:
    f.write(code)

print("Wrote api/ml_model.py")
