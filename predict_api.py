from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import traceback, joblib

def create_app():
    app = FastAPI(title="AutoML API", openapi_url="/openapi.json")

    class InputData(BaseModel):
        age: float
        balance: float
        duration: float
        campaign: float
        pdays: float
        previous: float

    @app.on_event("startup")
    def load_model():
        global model
        try:
            model = joblib.load("best_model.pkl")
            print("✅ Model loaded.")
        except Exception as e:
            print("❌ Model load failed:", e)
            model = None

    @app.get("/")
    def root():
        return {"message": "AutoML API running"}

    @app.post("/predict")
    def predict(data: InputData):
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded.")
        try:
            df = pd.DataFrame([data.dict()])
            pred = model.predict(df)[0]
            label = "yes" if pred == 1 else "no"
            return {"prediction": int(pred), "label": label}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    return app

app = create_app()
