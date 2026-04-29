from openai import OpenAI
from dotenv import load_dotenv
import json
import os

load_dotenv()

client = OpenAI()  # Reads OPENAI_API_KEY from .env automatically

prompt = """Create a Jupyter notebook as a valid JSON object in .ipynb format with 7 sections:
1. Overview: Load data, print shape/columns/dtypes, display first rows
2. Target Analysis: if classification, show class distribution (bar plot), if regression, show target histogram + stats
3. Missing Values: heatmap of nulls per column (seaborn.heatmap)
4. Feature Distributions: 3x3 subplot grid of histograms for all numeric features (use plt.subplots)
5. Correlation Matrix: heatmap of correlation between numeric features (annotated with values)
6. Features vs Target: 3 scatter plots (for regression) or box plots (for classification) showing strongest feature relationships with target
7. Key Findings: a markdown cell with 3-5 bullet points summarizing patterns discovered

Rules:
- Use matplotlib.pyplot and seaborn. Set figure sizes appropriately (e.g., figsize=(12, 8)). Title and label every plot.
- Return ONLY valid .ipynb JSON — no explanation, no markdown fences, no extra text.
- The notebook must have the standard structure with nbformat, nbformat_minor, metadata, and cells fields.
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}]
)

raw = response.choices[0].message.content

# Strip markdown fences if GPT wraps the JSON anyway
if raw.startswith("```"):
    raw = raw.split("```")[1]
    if raw.startswith("json"):
        raw = raw[4:]
    raw = raw.strip()

# Validate it's proper JSON before saving
notebook_json = json.loads(raw)

# Ensure the output folder exists
os.makedirs("notebooks", exist_ok=True)

# Save as a real .ipynb file
with open("notebooks/eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=2)

print("✅ File created successfully at notebooks/eda.ipynb")
