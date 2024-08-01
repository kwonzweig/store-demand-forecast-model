import pandas as pd


def preprocess(input_data):
    input_data = input_data.copy()
    input_data['date'] = pd.to_datetime(input_data['date'])

    input_data['month'] = input_data['date'].dt.month
    input_data['day'] = input_data['date'].dt.dayofweek
    input_data['year'] = input_data['date'].dt.year

    # Filter by feature columns only (Excluding id and date cols)
    filtered_cols = [col for col in input_data.columns if col not in ['date', 'id']]
    input_data = input_data[filtered_cols]

    return input_data
