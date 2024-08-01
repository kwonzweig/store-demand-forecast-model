import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_prep import preprocess


def pre_tuned_model(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, test_y: pd.Series) -> lgb.Booster:
    """
    Train a LightGBM model with pre-tuned parameters.

    Args:
        train_x (pd.DataFrame): Training features.
        train_y (pd.Series): Training target.
        test_x (pd.DataFrame): Validation features.
        test_y (pd.Series): Validation target.

    Returns:
        lgb.Booster: Trained LightGBM model.
    """
    params = {
        'nthread': 10,
        'max_depth': 5,
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'metric': 'mape',  # this is abs(a-e)/max(1,a)
        'num_leaves': 64,
        'learning_rate': 0.2,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 3.097758978478437,
        'lambda_l2': 2.9482537987198496,
        'verbose': 1,
        'min_child_weight': 6.996211413900573,
        'min_split_gain': 0.037310344962162616,
    }

    # Create datasets for training and validation
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_valid = lgb.Dataset(test_x, test_y)

    # Train the model
    model = lgb.train(
        params, lgb_train, 3000, valid_sets=[lgb_train, lgb_valid], early_stopping_rounds=50, verbose_eval=50
    )
    return model


def train_model(df_train: pd.DataFrame) -> lgb.Booster:
    """
    Preprocess the training data, split it into training and validation sets,
    and train the model.

    Args:
        df_train (pd.DataFrame): Training data.

    Returns:
        lgb.Booster: Trained LightGBM model.
    """
    # Preprocess the training data
    df_train_proc = preprocess(df_train)

    y_col = 'sales'
    feature_cols = [col for col in df_train_proc.columns if col not in [y_col]]

    # Split the data into training and validation sets
    train_x, test_x, train_y, test_y = train_test_split(
        df_train_proc[feature_cols], df_train_proc[y_col], test_size=0.2, random_state=2018
    )

    # Train the model with pre-tuned parameters
    model = pre_tuned_model(train_x, train_y, test_x, test_y)

    return model


def reprod_check(df_test: pd.DataFrame, model: lgb.Booster):
    """
    Check the reproducibility of the trained model by comparing predictions
    with a reference set of predictions.

    Args:
        df_test (pd.DataFrame): Test data.
        model (lgb.Booster): Trained LightGBM model.

    Returns:
        None
    """
    # Preprocess the test data
    df_test_proc = preprocess(df_test)
    # Generate predictions on the test data
    pred_test = model.predict(df_test_proc)

    # Load the reference predictions from a CSV file
    kaggle_note_pred = pd.read_csv('lgb_bayasian_param_kaggle.csv')

    # Compare the predictions for consistency
    consistency = np.allclose(pred_test, kaggle_note_pred['sales'].values)
    if not consistency:
        raise ValueError("Model predictions are inconsistent with the reference predictions.")
    print(f"Consistency between Kaggle Model and Production Model: {consistency}")


def main():
    """
    Main function to execute the training, testing, and saving of the model.
    """
    # Paths to the training and test data files
    train_data_path = os.path.join('..', 'data', 'train.csv')
    test_data_path = os.path.join('..', 'data', 'test.csv')

    # Load the training and test data
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)

    # Train the model
    model = train_model(df_train)

    # Check model reproducibility
    reprod_check(df_test, model)

    # Save the trained model to a file
    model.save_model('model.txt')
    print("Model trained and saved as model.txt")


if __name__ == "__main__":
    main()
