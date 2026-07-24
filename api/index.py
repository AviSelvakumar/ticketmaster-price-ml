"""Flask API for the minprice predictor, structured for Vercel's Python
serverless runtime.

Unlike app.py (the local/full-stack version), this uses ml_model.py -- a
dependency-free pure-Python export of the trained RandomForestRegressor
(via m2cgen) -- instead of loading model.pkl with scikit-learn/pandas/numpy.
That keeps this function's deployed size well under Vercel's serverless
function limit.

Regenerate ml_model.py by running train_model.py then export_model.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, jsonify, request, send_from_directory

from ml_model import score

BASE_DIR = Path(__file__).resolve().parent.parent
REQUIRED_FIELDS = {"weekend", "pop", "score", "month", "genre"}

# Order matches the OneHotEncoder's alphabetically-sorted categories_ used
# at training time (see api/ml_model.py generation in export_model.py),
# followed by the passthrough numeric columns weekend, pop, score, month.
GENRES = [
    "Classical", "Comedy", "Country", "Dance/Electronic", "Family", "Folk",
    "Hip-Hop/Rap", "Metal", "Other", "Pop", "R&B", "Religious", "Rock",
    "Undefined", "World",
]

app = Flask(__name__)


def validate_record(record):
    if not isinstance(record, dict):
        return "each record must be a JSON object"
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        return f"missing fields: {sorted(missing)}"
    for field in ("weekend", "pop", "score", "month"):
        try:
            float(record[field])
        except (TypeError, ValueError):
            return f"field '{field}' must be numeric"
    if not isinstance(record["genre"], str):
        return "field 'genre' must be a string"
    return None


def build_feature_vector(record):
    genre_onehot = [1.0 if record["genre"] == g else 0.0 for g in GENRES]
    numeric = [
        float(record["weekend"]),
        float(record["pop"]),
        float(record["score"]),
        float(record["month"]),
    ]
    return genre_onehot + numeric


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/genres")
def genres():
    return jsonify(genres=GENRES)


@app.get("/docs")
def docs():
    return send_from_directory(BASE_DIR / "static", "docs.html")


@app.get("/openapi.yaml")
def openapi_spec():
    return send_from_directory(BASE_DIR, "openapi.yaml", mimetype="application/yaml")


@app.post("/predict")
def predict():
    body = request.get_json(silent=True)
    if body is None:
        return jsonify(error="request body must be JSON"), 400

    records = body if isinstance(body, list) else [body]
    if not records:
        return jsonify(error="no records provided"), 400

    for i, record in enumerate(records):
        error = validate_record(record)
        if error:
            return jsonify(error=f"record {i}: {error}"), 400

    results = []
    for record in records:
        prediction = score(build_feature_vector(record))
        results.append({**record, "predicted_minprice": round(float(prediction), 2)})

    return jsonify(results=results if isinstance(body, list) else results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
