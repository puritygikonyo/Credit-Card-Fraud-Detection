from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  ## Load the .env file

client = OpenAI(api_key="*************")

prompt = """Create my-ml-project/data/data_cleaner that defines a clean_data(df) function. - Handles nulls: drop rows where target is null, forward-fill other columns (if time series) or drop rows (if not), drop columns > 50% nulls
- Removes duplicates: drop exact row duplicates, keep first
- Converts dtypes: ensure numeric columns are float/int, categorical columns are strings
- Saves cleaned CSV to data/cleaned.csv
- Re-runs the quality gate on cleaned data
- Returns cleaned df and quality result

Include __main__ block that loads raw data/, cleans it, prints before/after row counts and quality result.
"""
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}]
)

code = response.choices[0].message.content

# Automatically save the file into your data folder
with open("data/data_cleaner.py", "w", encoding="utf-8") as f:
    f.write(code)

print("✅ File created successfully at my-ml-project/data/data_cleaner.py") 


