"""Flask API for the minprice predictor -- local/full-stack version.

Named local_app.py (not app.py) deliberately: Vercel's Flask framework
detection treats a root-level app.py as THE serverless entrypoint,
overriding vercel.json's rewrite to api/index.py. That caused production
to import this file (which needs joblib/pandas/scikit-learn) even though
the deployed requirements.txt only installs Flask -- see api/index.py for
the dependency-free version that's actually meant to be deployed.

Run:
    python local_app.py

Endpoints:
    GET  /health           liveness check
    GET  /genres           genres the model was trained on
    POST /predict          predict minprice for one record or a list of records
    GET  /docs             ReDoc-rendered API documentation
    GET  /openapi.yaml     OpenAPI spec backing /docs
"""
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = "model.pkl"
REQUIRED_FIELDS = {"weekend", "pop", "score", "month", "genre"}
KNOWN_GENRES = [
    "Rock", "Hip-Hop/Rap", "Country", "R&B", "Metal", "Pop",
    "Dance/Electronic", "Undefined", "Comedy", "Other", "Folk",
    "Family", "World", "Religious", "Classical",
]

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


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


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/genres")
def genres():
    return jsonify(genres=KNOWN_GENRES)


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

    df = pd.DataFrame(records)[["weekend", "pop", "score", "month", "genre"]]
    predictions = model.predict(df)

    results = [
        {**records[i], "predicted_minprice": round(float(predictions[i]), 2)}
        for i in range(len(records))
    ]
    return jsonify(results=results if isinstance(body, list) else results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
