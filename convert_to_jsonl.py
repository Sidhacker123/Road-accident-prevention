
import pandas as pd
import json

# Load your dataset
df = pd.read_csv("cleaned.csv")

# Map severity numeric values to labels
severity_map = {1: "Slight", 2: "Serious", 3: "Fatal"}
df["Accident_severity"] = df["Accident_severity"].map(severity_map)

# Drop rows with missing key data
columns = [
    "Weather_conditions", "Light_conditions", "Road_surface_type",
    "Types_of_Junction", "Cause_of_accident", "Accident_severity"
]
df = df[columns].dropna()

# Format as prompt-completion pairs
def to_prompt_completion(row):
    prompt = (
        f"Weather: {row['Weather_conditions']}. "
        f"Light: {row['Light_conditions']}. "
        f"Road surface: {row['Road_surface_type']}. "
        f"Junction: {row['Types_of_Junction']}. "
        f"Cause: {row['Cause_of_accident']}. "
        f"What is the accident severity?"
    )
    completion = f" {row['Accident_severity']}"
    return {"prompt": prompt, "completion": completion}

with open("full_fine_tune_data.jsonl", "w") as f:
    for _, row in df.iterrows():
        json.dump(to_prompt_completion(row), f)
        f.write("\n")
