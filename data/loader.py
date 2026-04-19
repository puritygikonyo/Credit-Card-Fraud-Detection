#To accomplish your requirements, you'll want to create a Python script that uses libraries such as `pandas` to handle and analyze the CSV data file. Below is a sample implementation of `loader.py` that fulfills all the specified tasks:


import pandas as pd
import os

def load_csv(filename):
    """Load a CSV file from the data directory."""
    file_path = os.path.join('data', 'creditcard.csv')
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"The file {filename} does not exist in the data directory.")
        return None

def print_shape(df):
    """Print the shape of the DataFrame."""
    print(f"Shape (rows, columns): {df.shape}")

def print_column_info(df):
    """Print column names and their data types."""
    print("\nColumn names and data types:")
    print(df.dtypes)

def print_summary_statistics(df):
    """Print summary statistics for numerical columns."""
    print("\nSummary statistics for numerical columns:")
    print(df.describe())

def print_missing_values(df):
    """Print missing value counts and percentages for each column."""
    print("\nMissing values count and percentage:")
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    missing_info = pd.DataFrame({'Count': missing_counts, 'Percentage': missing_percentages})
    print(missing_info)

def main():
    # Specify the filename of the CSV you wish to load
    filename = 'creditcard.csv'  # Replace with your actual file name
    
    # Load the CSV
    df = load_csv(filename)
    
    if df is not None:
        # Perform the analysis
        print_shape(df)
        print_column_info(df)
        print_summary_statistics(df)
        print_missing_values(df)

if __name__ == "__main__":
    main()


### How to use this script:

#1. **Place your CSV file** in the `data/` directory right next to your `loader.py` script. Make sure to update the `filename` variable in the `main()` function with the correct name of your CSV file.

#2. **Run the script** by executing the following command in your terminal:
##`` python my-ml-project/data/loader.py ```

#This script will load the specified CSV file, print the shape, column names with data types, summary statistics for numerical columns, and missing value counts and percentages. Make sure you have the `pandas` library installed by running `pip install pandas` if necessary.

