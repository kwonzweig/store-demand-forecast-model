from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import lightgbm as lgb
import pandas as pd
import datetime

from model.data_prep import preprocess

app = FastAPI()


# Define a Pydantic model for the request body
class PredictionRequest(BaseModel):
    date: str
    store: int
    item: int


# Load the trained model
model = lgb.Booster(model_file='model.txt')


# Define the /predict endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    # Parse the date and create features
    try:
        date = pd.to_datetime(request.date)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    data = {
        "store": request.store,
        "item": request.item,
        "month": date.month,
        "day": date.dayofweek,
        "year": date.year
    }

    df = pd.DataFrame([data])

    df_proc = preprocess(df)

    # Make a prediction
    try:
        prediction = model.predict(df_proc)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Model prediction failed.")

    # Return the prediction as a response
    return {"sales": prediction[0]}


# Define the /status endpoint
@app.get("/status")
def status():
    return {"status": "API is running"}
