######################################
##Prompt 2 to built a data quality gate
from openai import OpenAI 
from dotenv import load_dotenv

load_dotenv()  # loads the .env file

client = OpenAI(api_key="*********************")

prompt = """Create my-ml-project/data/data_quality_gate that defines a check_data_quality(df) function. It should run 5 checks and return a dict with structure:
                {
                    'success': bool (all checks passed),
                    'failures': list of critical errors (missing required columns, etc),
                    'warnings': list of concerns (high null rate, low variance. etc),
                    'statistics': dict with counts (total_rows, total_nulls_by_column, etc)
                }

                The 5 checks are:
                1. Schema validation: required columns exist + correct dtypes
                2. Row count: at least 100 rows, warn if < 1000
                3. Null rates: no column > 50% nulls (critical), warn if > 20%
                4. Value ranges: numeric columns with sensible bounds (e.g., no negative counts, no 10000% values)
                5. Target distribution: if classification, target must have 2+ classes and each >= 5% of data (warn if imbalanced)


                Include a_main_block that loads data/ CSV and runs the gate, prints result.

                """
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}]
)

code = response.choices[0].message.content

# Automatically save the file into your data folder
with open("data/data_quality_gate.py", "w", encoding="utf-8") as f:
    f.write(code)

print("✅ File created successfully at my-ml-project/data/data_quality_gate.py") 

