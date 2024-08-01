import os

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model.data_prep import preprocess

app = FastAPI()


class PredictionRequest(BaseModel):
    """
    Pydantic model for the prediction request body.

    Attributes:
        date (str): The date of the prediction. Can be in multiple formats
            (e.g., "YYYY-MM-DD", "MM/DD/YYYY", "DD-MM-YYYY").
        store (int): The store identifier.
        item (int): The item identifier.
    """

    date: str
    store: int
    item: int


class PredictionResponse(BaseModel):
    """
    Pydantic model for the prediction response.

    Attributes:
        sales (float): The predicted sales value.
    """

    sales: float


# Load the trained LightGBM model
model = lgb.Booster(model_file=os.path.join("model", "model.txt"))


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict sales based on the provided store, item, and date.

    Args:
        request (PredictionRequest): The prediction request containing date, store, and item.

    Returns:
        PredictionResponse: The response containing the predicted sales.

    Raises:
        HTTPException: If the date format is invalid or model prediction fails.
    """
    # Parse the date and create features
    try:
        date = pd.to_datetime(request.date)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Please provide a valid date string.",
        )

    # Prepare the data for prediction
    data = {
        "store": request.store,
        "item": request.item,
        "date": date,
    }

    df = pd.DataFrame([data])

    # Preprocess the data
    df_proc = preprocess(df)

    # Make a prediction using the model
    try:
        prediction = model.predict(df_proc)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Model prediction failed.")

    # Return the prediction as a response
    return PredictionResponse(sales=prediction[0])


@app.get("/status")
def status() -> dict:
    """
    Check the status of the API.

    Returns:
        dict: A dictionary indicating that the API is running.
    """
    return {"status": "API is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
