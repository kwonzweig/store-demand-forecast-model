import os

import numpy as np
import pandas as pd
import pytest
from lightgbm import Booster

from model.data_prep import preprocess
from model.train import train_model


@pytest.fixture(scope="module")
def trained_model():
    """
    Fixture to train and provide the LightGBM model.

    Returns:
        Booster: Trained LightGBM model.
    """
    # Load the training data
    train_data_path = os.path.join("data", "train.csv")
    df_train = pd.read_csv(train_data_path)

    # Train the model
    model = train_model(df_train)

    return model


@pytest.fixture(scope="module")
def test_data():
    """
    Fixture to provide the test data.

    Returns:
        pd.DataFrame: Test data.
    """
    test_data_path = os.path.join("data", "test.csv")
    return pd.read_csv(test_data_path)


def test_reprod_check(trained_model: Booster, test_data: pd.DataFrame):
    """
    Test the reproducibility check function.

    Args:
        trained_model (Booster): The trained LightGBM model.
        test_data (pd.DataFrame): The test data.
    """
    # Path to the reference prediction CSV file
    reference_pred_path = os.path.join("model", "lgb_bayasian_param.csv")

    # Check if the reference prediction file exists
    assert os.path.exists(reference_pred_path), "Reference prediction file not found."

    # Preprocess the test data
    df_test_proc = preprocess(test_data)

    # Generate predictions on the test data
    pred_test = trained_model.predict(df_test_proc)

    # Load the reference predictions
    kaggle_note_pred = pd.read_csv(reference_pred_path)

    # Ensure the reference predictions are loaded correctly
    assert (
        "sales" in kaggle_note_pred.columns
    ), "Reference predictions must contain 'sales' column."

    # Compare the predictions for consistency
    consistency = np.allclose(pred_test, kaggle_note_pred["sales"].values)
    assert (
        consistency
    ), "Model predictions are inconsistent with the reference predictions."
    print(f"Reproducibility from Kaggle Model to Production Model: {consistency}")
