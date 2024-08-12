import pandas as pd
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from tqdm import tqdm
import csv
import os

# Load the model and tokenizer
hf_token = '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'

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model = LlamaForCausalLM.from_pretrained(
    model_name,
    token=hf_token, 
    torch_dtype=torch.float16, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

# Setup the pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Read the input CSV file
df = pd.read_csv('captions/sat_captions_50k.csv')

# Add a new column for summarization if it doesn't exist
if 'Summarization' not in df.columns:
    df['Summarization'] = ""


# Output CSV file path
output_dir = 'summarizations/llama-3-8b-instruct'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'summarizations-llama-3-8b-instruct_subset.csv')

# Determine the set of already processed SatelliteIDs
processed_satellite_ids = set()
try:
    with open(output_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            processed_satellite_ids.add(int(row[0]))  # Assuming SatelliteID is the first column
except FileNotFoundError:
    # Write the header to the output CSV file if it doesn't exist
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(df.columns)
        
df = df[~df['SatelliteID'].isin(processed_satellite_ids)]
df = df[df['GroundImageIDs'].str.count(',') > 0]

count = 0 
# Process each row in the dataframe
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    # satellite_id = row['SatelliteID']
  
    # # Skip if the satellite image has been processed
    # if satellite_id in processed_satellite_ids:
    #     continue

    captions = row['Captions']
    prompt = (
        f"The texts in the bracket are descriptions of images taken in the same region. Summarize the text to describe the region."
        f"[{captions}]"
        " <Summarization>:"
    )
    print(prompt)
    # Generate the summarization
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=200,
    )

    # Extract the generated text
    text = []
    for seq in sequences:
        generated_text = seq['generated_text']
        if "<Summarization>:" in generated_text:
            summarization = generated_text.split("<Summarization>:")[1].strip()
            text.append(summarization)
        else:
            text.append(generated_text.strip())

    # Update the dataframe with the summarization
    df.at[index, 'Summarization'] = " ".join(text)

    # Append the updated row to the output CSV file
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(df.loc[index].values)

    # Add the processed SatelliteID to the set
    # processed_satellite_ids.add(satellite_id)

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Optional: break after a certain number of rows for testing
    count += 1
    if count == 10:
        break
