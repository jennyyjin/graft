import pandas as pd
import json
from tqdm import tqdm
import random
import re

# List of CSV files to load
csv_files = [
    '../graft/summarizations/llama-3-8b-instruct/summarizations-llama-3-8b-instruct_batched_naip_0_250_grid.csv',
    '../graft/summarizations/llama-3-8b-instruct/summarizations-llama-3-8b-instruct_batched_naip_6_cont_grid.csv',
    '../graft/summarizations/llama-3-8b-instruct/summarizations-llama-3-8b-instruct_batched_naip_251_475_grid.csv'
]

# Load all CSV files into one DataFrame
df_list = []
for file in csv_files:
    df_chunk = pd.read_csv(file)
    df_list.append(df_chunk)

sum_csv = pd.concat(df_list, ignore_index=True)

# Load the JSON file with the conversations to be processed (replace 'json_file_path' with actual path)
json_file_path = 'merged_data_11_5.json'
with open(json_file_path, 'r') as f:
    json_data = json.load(f)

# Extract relevant columns from the combined CSV (adjust column names if necessary)
image_id_column = 'SatelliteID'  # Adjust if necessary

# Function to find the image path by image ID
def find_image_path(image_id):
    result = sum_csv[sum_csv[image_id_column].astype(str).str.contains(str(image_id), na=False)]
    if not result.empty:
        # print(f"Found image path for image ID {image_id}: {result[image_id_column].values[0]}")
        return result[image_id_column].values[0]  # Adjust if necessary
    else:
        print(f"Image path not found for image ID {image_id}")
        return None

pattern = r'\s*\n\s*<image>$'
# Process only the first JSON file
# for entry in tqdm(json_data, desc="Processing entries"):
# Collect indices to remove after processing
entries_to_remove = []

# Process only the first JSON file
for i, entry in tqdm(enumerate(json_data)):
    entry['id'] = str(entry['id']).zfill(8)  # Ensure image ID has leading zeros if necessary
    
    # Check if conversations are empty
    if not entry['conversations']:
        entries_to_remove.append(i)
        continue
    
    image_id = str(entry['id'])  # Assuming 'id' in the JSON is the satellite ID or a reference to it
    new_image_path = find_image_path(image_id)
    
    if new_image_path:
        entry['image'] = "../../.." + new_image_path  # Replace the image path with the new one
    else:
        print(f"Image path not found for image ID {image_id}")
        entries_to_remove.append(i)
        continue

    # Process conversations: limit to 5 pairs if more than 10
    if len(entry['conversations']) > 10:
        new_conversations = []
        num_pairs = len(entry['conversations']) // 2  # Number of conversation pairs
        selected_pairs = random.sample(range(num_pairs), 5)  # Randomly choose 5 pairs

        for idx in selected_pairs:
            new_conversations.append(entry['conversations'][2 * idx])  # Append human value
            new_conversations.append(entry['conversations'][2 * idx + 1])  # Append GPT value

        entry['conversations'] = new_conversations

    # Ensure only the first human question retains the <image> token
    image_token_retained = False
    for conversation in entry['conversations']:
        if conversation['from'] == 'human':
            if not image_token_retained:
                # Handle <image> token positioning
                if re.search(pattern, conversation['value']):
                    if random.choice([True, False]):
                        conversation['value'] = re.sub(pattern, "", conversation['value']).strip()
                        conversation['value'] = f"<image>\n{conversation['value']}"
                    else:
                        conversation['value'] = re.sub(pattern, "\n<image>", conversation['value']).strip()
                image_token_retained = True
            else:
                conversation['value'] = re.sub(pattern, "", conversation['value']).replace("<image>", "").strip()

# Now remove the entries that were flagged for removal
for idx in sorted(entries_to_remove, reverse=True):
    json_data.pop(idx)


# # # Load the second JSON file (already processed, so it won't be altered)
# second_json_file_path = '../graft_vqa/playground/data/processed_conversations_updated_imagetoken.json'  # Replace with actual path
# with open(second_json_file_path, 'r') as f:
#     second_json_data = json.load(f)

# # # Concatenate the content of the second JSON file without processing
# json_data.extend(second_json_data)

# Save the updated JSON file with concatenated data
updated_json_file_path = '../graft_vqa/playground/data/processed_complex_reasoning_referring_detailed_description_11_5.json'
with open(updated_json_file_path, 'w') as f:
    json.dump(json_data, f, indent=2)

print("First JSON processed, second JSON concatenated, and final JSON saved.")
