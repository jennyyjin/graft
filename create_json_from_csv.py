import pandas as pd
import json
from tqdm import tqdm

# Load the CSV files
file1 = 'ins_gen_referring_pixel.csv'
file2 = 'ins_gen_complex_reasoning.csv'
file3 = 'ins_gen_detailed_description.csv'

# Read each CSV file into a DataFrame
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

# Merge the DataFrames, assuming they have the same columns
merged_df = pd.concat([df1, df2, df3], ignore_index=True)

# Function to process each row into the desired JSON structure
def process_row(row):
    # Initialize the JSON structure
    item = {
        "id": str(row["sat_id"]).zfill(8),
        "image": f"../../../share/hariharan/ukm4/CLIPRS10m_m2o/naipimages/0001/{str(row['sat_id']).zfill(8)}.jpg",
        "conversations": []
    }
    
    # Check if there are questions and answers in the 'instructions' column
    if "Question:" in row["instructions"] and "Answer:" in row["instructions"]:
        # Split the instructions into individual question-answer pairs
        qa_pairs = row["instructions"].split("Question:")
        for pair in qa_pairs[1:]:  # Skip the first split as it will be empty
            question, answer = pair.split("Answer:", 1)
            item["conversations"].append({
                "from": "human",
                "value": f"{question.strip()}\n<image>"
            })
            item["conversations"].append({
                "from": "gpt",
                "value": answer.strip()
            })
    else:
        # Use the detailed description if no questions and answers are present
        item["conversations"].append({
            "from": "human",
            "value": "Describe the image in detail\n<image>"
        })
        item["conversations"].append({
            "from": "gpt",
            "value": row["instructions"]
        })

    return item

# Apply the function to each row of the DataFrame
json_data = [process_row(row) for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Processing rows")]

# Save the data to a JSON file
output_file = 'merged_data_11_5.json'
with open(output_file, 'w') as f:
    json.dump(json_data, f, indent=2)

print(f"JSON file has been saved to {output_file}")
