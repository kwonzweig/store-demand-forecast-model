
# Forecast Model API

This page provides instructions on how to run the API.

#### For details on the approach, please check the [project document](https://salt-cylinder-2c5.notion.site/Store-Demand-Forecasting-Model-API-1b87af2eb0314353bdde03df9583a765?pvs=4).

## Overview
- **Objectives**: Prediction of `sales` for a given `store` and `item`.
- **API Framework**: `FastAPI`
- **Training Algorithm**: `LightGBM`

## Getting Started

### Prerequisites
- Python 3.6.4
- Docker (for containerization)

### Running API with Docker

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kwonzweig/store-demand-forecast-model.git
   cd store-demand-forecast-model
   ```

2. **Build and run the API**:
   ```bash
   docker build -t forecast-model-api .
   docker run -p 8000:8000 forecast-model-api
   ```

   Local API endpoint: `http://localhost:8000`

### Running API without Docker

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API locally**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## API Endpoints

- **`/predict`** (POST): Predict the number of sales.
  - **Request Body**:
    ```json
    {
      "date": "2013-01-01",
      "store": 1,
      "item": 1
    }
    ```
  - **Response**:
    ```json
    {
      "sales": 9.053297510226828
    }
    ```

- **`/status`** (GET): Check the status of the API.
  - **Response**:
    ```json
    {
      "status": "API is running"
    }
    ```

## Training the Model

A pretrained model is already provided as model.pkl, but you can run the training script to update the model registry.

1. **Prepare the dataset**:

   Place the dataset (`train.csv`, `test.csv`) in the `data/` directory if not already present.

2. **Train the model**:
   ```bash
   python model/train.py
   ```
   This will train the model and save it as `model/model.pkl`.
