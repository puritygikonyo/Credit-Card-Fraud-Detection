from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # loads the .env file

client = OpenAI(api_key="***********")

prompt = """Create my-ml-project/data/loader.py that:
- Loads the CSV from data/
- Prints the shape (rows, columns)
- Prints column names and data types
- Prints summary statistics (mean, std, min, max for numerics)
- Prints missing value counts and percentages
- Has a main block that runs all of the above when executed"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}]
)

code = response.choices[0].message.content

# Automatically save the file into your data folder
with open("data/loader.py", "w", encoding="utf-8") as f:
    f.write(code)

print("✅ File created successfully at my-ml-project/data/loader.py") 

