import pandas as pd


def preprocess(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the input data by extracting date components and filtering columns.

    Args:
        input_data (DataFrame): The input data containing a 'date' column and other features.

    Returns:
        DataFrame: The preprocessed data with additional date-related features and filtered columns.
    """
    # Create a copy of the input data to avoid modifying the original data
    data_proc = input_data.copy()

    # Convert the 'date' column to datetime format
    data_proc["date"] = pd.to_datetime(data_proc["date"])

    # Print date range
    print(f"Date range: {data_proc['date'].min()} to {data_proc['date'].max()}")

    # Extract month, day of the week, and year from the 'date' column
    data_proc["month"] = data_proc["date"].dt.month
    data_proc["day"] = data_proc["date"].dt.dayofweek
    data_proc["year"] = data_proc["date"].dt.year

    # Filter out 'date' and 'id' columns, keeping only feature columns
    filtered_cols = [col for col in data_proc.columns if col not in ["date", "id"]]
    data_proc = data_proc[filtered_cols]

    # Print list of stores & items pairs
    store_item_pairs = data_proc.groupby(["store", "item"]).size().reset_index(name="count")
    print("Store-Item pairs:")
    print(store_item_pairs)

    return data_proc
