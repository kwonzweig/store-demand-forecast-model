import time

import pytest
from fastapi.testclient import TestClient

from app.main import app

# Create a TestClient instance for the FastAPI app
client = TestClient(app)


@pytest.fixture(scope="module")
def test_client():
    """
    Create a TestClient for testing.

    Returns:
        TestClient: The test client for the FastAPI app.
    """
    return TestClient(app)


def test_status(test_client):
    """
    Test the /status endpoint.

    Args:
        test_client (TestClient): The test client for the FastAPI app.
    """
    response = test_client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running"}


def test_predict_valid(test_client):
    """
    Test the /predict endpoint with valid input data.

    Args:
        test_client (TestClient): The test client for the FastAPI app.
    """
    # Define a valid request body
    request_body = {"date": "2024-07-31", "store": 1, "item": 1}

    response = test_client.post("/predict", json=request_body)
    assert response.status_code == 200
    assert "sales" in response.json()
    assert isinstance(response.json()["sales"], float)


def test_predict_invalid_date(test_client):
    """
    Test the /predict endpoint with an invalid date format.

    Args:
        test_client (TestClient): The test client for the FastAPI app.
    """
    # Define a request body with an invalid date
    request_body = {"date": "31-07-202", "store": 1, "item": 1}

    response = test_client.post("/predict", json=request_body)
    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "Invalid date format. Please provide a valid date string."
    )


def test_predict_missing_fields(test_client):
    """
    Test the /predict endpoint with missing fields in the request body.

    Args:
        test_client (TestClient): The test client for the FastAPI app.
    """
    # Define a request body with missing fields
    request_body = {
        "date": "2024-07-31",
        "store": 1
        # 'item' field is missing
    }

    response = test_client.post("/predict", json=request_body)
    assert response.status_code == 422  # Unprocessable Entity
    assert response.json()["detail"][0]["msg"] == "field required"


def test_predict_out_of_range(test_client):
    """
    Test the /predict endpoint with date, store and item IDs far outside the training set.

    Args:
        test_client (TestClient): The test client for the FastAPI app.
    """
    # Define a request body with out-of-range store and item
    request_body = {
        "date": "1800-07-31",
        "store": 100,  # Store ID not in training set
        "item": 1000,  # Item ID not in training set
    }

    response = test_client.post("/predict", json=request_body)
    assert response.status_code == 200
    assert "sales" in response.json()
    assert isinstance(response.json()["sales"], float)


def test_predict_int_date(test_client):
    """
    Test the /predict endpoint with a request body containing an integer date
    to see if FastAPI coerce the int into str properly.


    Args:
        test_client (TestClient): The test client for the FastAPI app.
    """
    # Define a request body with the wrong format
    request_body = {
        "date": 20240731,  # Date is an integer
        "store": 1,
        "item": 1,
    }

    response = test_client.post("/predict", json=request_body)
    assert response.status_code == 200
    assert "sales" in response.json()
    assert isinstance(response.json()["sales"], float)


def test_predict_performance(test_client):
    """
    Test the performance of the /predict endpoint with valid input data.

    Args:
        test_client (TestClient): The test client for the FastAPI app.
    """
    # Define a valid request body
    request_body = {
        "date": "2024-07-31",
        "store": 1,
        "item": 1
    }

    # Measure response time
    start_time = time.time()
    response = test_client.post("/predict", json=request_body)
    end_time = time.time()
    duration = end_time - start_time

    # Check that the response time is within an acceptable limit
    response_time_limit = 5.0
    assert duration < response_time_limit, f"Response time too long: {duration} seconds"
    assert response.status_code == 200
    assert "sales" in response.json()
    assert isinstance(response.json()["sales"], float)


def test_predict_invalid_data_types(test_client):
    """
    Test the /predict endpoint with invalid data types for store and item.

    Args:
        test_client (TestClient): The test client for the FastAPI app.
    """
    # Define a request body with string data types for store and item
    request_body = {
        "date": "2024-07-31",
        "store": "one",
        "item": "two"
    }

    response = test_client.post("/predict", json=request_body)
    assert response.status_code == 422  # Unprocessable Entity
    assert response.json()["detail"][0]["msg"] == "value is not a valid integer"
