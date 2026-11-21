"""API de prédiction AgriPredict.

Lancer depuis la racine du projet :
    uvicorn api.main:app --reload
Puis ouvrir http://127.0.0.1:8000/docs
"""

import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="AgriPredict API")

model = joblib.load("models/model.joblib")
with open("models/meta.json") as f:
    meta = json.load(f)


class Inputs(BaseModel):
    crop: str
    year: int
    temp_anomaly: float
    rainfall: float
    fertilizer: float
    pesticides: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta")
def get_meta():
    return meta


@app.post("/predict")
def predict(x: Inputs):
    if x.crop not in meta["crops"]:
        raise HTTPException(
            status_code=400,
            detail=f"Culture inconnue. Choix possibles : {meta['crops']}",
        )
    y = float(model.predict(pd.DataFrame([x.model_dump()]))[0])
    return {"crop": x.crop, "year": x.year, "yield_t_ha": round(y, 3)}
