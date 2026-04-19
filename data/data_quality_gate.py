#To create a data quality gate, you will first need to define the directory structure and create a Python script that contains the `check_data_quality` function. This script will include the checks specified and print the results when run. Below is an example of how you can set up this project:

##
#my-ml-project/
#└── data/
#    ├── data_quality_gate.py
#    └── creditcard.csv
##

#Here's the implementation of the `data_quality_gate.py` file:


import pandas as pd
import numpy as np

def check_data_quality(df):
    # Define the requirements for schema validation
    required_columns = {
        'Time': np.float64,
        'V1': np.float64,
        'V2': np.float64,
        'V3': np.float64,
        'V4': np.float64,
        'V5': np.float64,
        'V6': np.float64,
        'V7': np.float64,
        'V8': np.float64,
        'V9': np.float64,
        'V10': np.float64,
        'V11': np.float64,
        'V12': np.float64,
        'V13': np.float64,
        'V14': np.float64,
        'V15': np.float64,
        'V16': np.float64,
        'V17': np.float64,
        'V18': np.float64,
        'V19': np.float64,
        'V20': np.float64,
        'V21': np.float64,
        'V22': np.float64,
        'V23': np.float64,
        'V24': np.float64,
        'V25': np.float64,
        'V26': np.float64,
        'V27': np.float64,
        'V28': np.float64,
        'Amount': np.float64,
        'Class': np.int64   }

    # Initialize the result dictionary
    results = {
        'success': True,
        'failures': [],
        'warnings': [],
        'statistics': {}
    }
    
    # 1. Schema validation: Check required columns exist with correct dtypes
    for column, dtype in required_columns.items():
        if column not in df.columns:
            results['failures'].append(f'Missing required column: {column}')
            results['success'] = False
        elif not pd.api.types.is_dtype_equal(df[column].dtype, dtype):
            results['failures'].append(f'Incorrect dtype for column {column}: Expected {dtype}, got {df[column].dtype}')
            results['success'] = False
    
    # 2. Row count check
    total_rows = df.shape[0]
    results['statistics']['total_rows'] = total_rows
    if total_rows < 100:
        results['failures'].append('Row count is less than 100.')
        results['success'] = False
    elif total_rows < 1000:
        results['warnings'].append('Row count is less than 1000.')

    # 3. Null rates check
    null_rates = df.isnull().mean()
    results['statistics']['total_nulls_by_column'] = df.isnull().sum().to_dict()
    for column, null_rate in null_rates.items():
        if null_rate > 0.50:
            results['failures'].append(f'Column {column} has more than 50% null values.')
            results['success'] = False
        elif null_rate > 0.20:
            results['warnings'].append(f'Column {column} has more than 20% null values.')

    # 4. Value ranges check
    pca_columns = [f'V{i}' for i in range(1, 29)]  # V1 to V28
    for column in df.select_dtypes(include=[np.number]).columns:
        if column not in pca_columns:  # Skip V1-V28 as they naturally have negative values
            if (df[column] < 0).any():
                results['failures'].append(f'Negative values found in {column}')
                results['success'] = False
        if column not in ['Time', 'Amount']:  # Skip Time and Amount for large value check
            if (df[column] > 10000).any():
                results['warnings'].append(f'Unusually large values found in {column}.')
    # 5. Target distribution check for classification
    if 'Class' in df.columns:
        class_counts = df['Class'].value_counts(normalize=True)
        if len(class_counts) < 2:
            results['failures'].append('Target column has fewer than 2 classes.')
            results['success'] = False
        elif (class_counts < 0.05).any():
            results['warnings'].append('Target column classes with < 5% of data detected indicating imbalance.')

    return results

if __name__ == "__main__":
    # Load data from CSV
    try:
        df = pd.read_csv('data/creditcard.csv')
    except FileNotFoundError:
        raise Exception("CSV file not found. Please ensure your data is in 'your_data.csv' in the 'data/' directory.")

    # Run data quality gate
    results = check_data_quality(df)
    
    # Print results
    print("Data Quality Check Results:")
    print("Success:", results['success'])
    print("Failures:", results['failures'])
    print("Warnings:", results['warnings'])
    print("Statistics:", results['statistics'])


### Instructions:
#- Replace `'example_column_1'` and `'example_column_2'` with the actual columns and expected data types in your dataset.
#- Make sure your data is saved as `'your_data.csv'` in the `data/` directory.

#When you run `data_quality_gate.py`, it will read your CSV file and perform the checks, printing the results to the console.

