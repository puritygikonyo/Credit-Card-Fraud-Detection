#Certainly! Below is a possible implementation of your `my-ml-project/data/data_cleaner.py` script with the specified functionality. This script is designed to handle nulls, remove duplicates, convert data types, and save the cleaned dataset while running a quality gate on the cleaned data.


import os
import pandas as pd

def clean_data(df, target_column, time_series=False):
    # Handle nulls
    # Drop rows where the target is null
    df = df.dropna(subset=['Class'])
    
    if time_series:
        # Forward-fill other columns if time series
        df = df.fillna(method='ffill')
    else:
        # Drop rows where any other column is null
        df = df.dropna()
    
    # Drop columns with more than 50% null values
    threshold = len(df) * 0.5
    df = df.dropna(axis=1, thresh=threshold)
    
    # Remove duplicates
    df = df.drop_duplicates(keep='first')
    
    # Convert dtypes
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], errors='coerce')

        if pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            df[column] = df[column].astype(str)
    
    # Save cleaned data
    df.to_csv('data/cleaned.csv', index=False)
    
    # Re-run quality gate on cleaned data
    quality_result = run_quality_gate(df)
    
    return df, quality_result

def run_quality_gate(df):
    # Example quality gate check: ensure no remaining nulls and columns with >50% nulls
    null_check = df.isnull().sum().sum() == 0
    sufficient_data_check = all(df.count() >= len(df) / 2)

    # Further quality checks can be added here
    return {
        'null_check': null_check,
        'sufficient_data_check': sufficient_data_check
    }

if __name__ == "__main__":
    # Load raw data from file
    raw_data_path = 'data/creditcard.csv'  # Modify path as necessary
    df_raw = pd.read_csv(raw_data_path)

    # Define necessary parameters
    target_column = 'Class'  # Replace with your target column
    time_series = False  # Set True if the data is time series
    
    # Print initial row count
    print(f"Initial number of rows: {len(df_raw)}")
    
    # Clean the data
    df_cleaned, quality_result = clean_data(df_raw, target_column, time_series)
    
    # Print cleaned row count and quality result
    print(f"Number of rows after cleaning: {len(df_cleaned)}")
    print("Quality Result:", quality_result)


### Instructions
##1. **Adjust File Paths and Column Names:** You must modify the file paths (`raw_data_path`) and target column name (`target_column`) to fit your specific dataset.
##2. **Install Required Libraries:** Ensure you have `pandas` installed in your Python environment.
##3. **Time Series:** Set the `time_series` flag to `True` if you are dealing with a time series dataset, as this will forward-fill missing values instead of dropping them.

