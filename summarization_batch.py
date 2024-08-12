import pandas as pd
import torch
import transformers
from transformers import LlamaForCausalLM, AutoTokenizer
from tqdm import tqdm
import csv
import os
print("Imported libraries")

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
# df = pd.read_csv('captions/sat_captions_160k.csv')
# Load the CSV file
file_path = 'captions/combined_naip.csv'
data = pd.read_csv(file_path)

# Define a function to merge rows with the same SatelliteID and concatenate the captions
def merge_and_concatenate(df):
    # Group by SatelliteID
    grouped = df.groupby('Satellite Image Path')
    
    # Function to concatenate captions
    def concatenate_data(group):
        # captions = "\n".join([f"{i+1}. {caption}" for i, caption in enumerate(group['Caption'])])
        captions = [caption.replace('\n', ' ') for caption in group['Caption']]
        captions = "\n".join([f"{i+1}. {caption}" for i, caption in enumerate(captions)])
        ground_image_ids = ", ".join(group['Ground Image Path'].astype(str))
        return pd.Series({
            'SatelliteID': group.name,
            'Captions': captions,
            'GroundImageIDs': ground_image_ids
        })
    
    # Apply the function to each group and reset the index
    merged_data = grouped.apply(concatenate_data).reset_index(drop=True)
    
    return merged_data

# Apply the function to the data
df = merge_and_concatenate(data)

# Add a new column for summarization if it doesn't exist
if 'Summarization' not in df.columns:
    df['Summarization'] = ""

# Output CSV file path
output_dir = 'summarizations/llama-3-8b-instruct'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'summarizations-llama-3-8b-instruct_batched_naip.csv')

# Determine the set of already processed SatelliteIDs
# processed_satellite_ids = set()
# try:
#     with open(output_file, 'r', newline='') as csvfile:
#         reader = csv.reader(csvfile)
#         next(reader)  # Skip header
#         for row in reader:
#             processed_satellite_ids.add(int(row[0]))  # Assuming SatelliteID is the first column
# except FileNotFoundError:
#     # Write the header to the output CSV file if it doesn't exist
#     with open(output_file, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(df.columns)

# Filter out already processed rows
# df = df[~df['SatelliteID'].isin(processed_satellite_ids)]
# df = df[df['GroundImageIDs'].str.count(',') == 0]

try:
    output_df = pd.read_csv(output_file)
    processed_data = output_df.set_index('SatelliteID')['GroundImageIDs'].to_dict()
    def needs_processing(satellite_id, ground_image_ids):
        if satellite_id in processed_data:
            return ground_image_ids.count(',') != processed_data[satellite_id].count(',')
        return True

    filtered_df = df[df.apply(lambda row: needs_processing(row['SatelliteID'], row['GroundImageIDs']), axis=1)]
    print(f"Filtered out {len(df) - len(filtered_df)} already processed rows")
except FileNotFoundError:
    # If the output file doesn't exist, create an empty dictionary
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(df.columns)
    filtered_df = df

batch_size = 16  # Define batch size
prompts = []
rows_to_process = []

# Create prompts and batches
for index, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Processing rows"):
    captions = row['Captions']
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant for summarizing image descriptions from the same region."},
        {"role": "user", "content": f"The texts in the brackets are descriptions of images taken in the same region. Summarize these descriptions concisely to provide an overview of the region. Be brief and avoid unnecessary preambles. [{captions}] <Summary>:"}
    ]
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompts.append(prompt)
    rows_to_process.append(index)
    
    # Process batch
    if len(prompts) == batch_size:
        with tqdm(total=len(prompts), desc="Processing batch") as pbar:
            sequences = pipeline(
                prompts,
                max_new_tokens=200,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            for i, batch in enumerate(sequences):
                for seq in batch:
                    generated_text = seq['generated_text']
                    if "<Summary>:" in generated_text:
                        summarization = generated_text.split("<Summary>:")[1].strip()
                        summarization = summarization.split("\n\n", 1)[-1].strip()  # Extract part after "assistant\n\n"
                        df.at[rows_to_process[i], 'Summarization'] = summarization
                    else:
                        df.at[rows_to_process[i], 'Summarization'] = generated_text.strip()
                pbar.update(1)

        # Append the updated rows to the output CSV file
        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for idx in rows_to_process:
                writer.writerow(df.loc[idx].values)
        
        # Clear lists and GPU cache
        prompts.clear()
        rows_to_process.clear()
        torch.cuda.empty_cache()

# Process any remaining rows
if prompts:
    with tqdm(total=len(prompts), desc="Processing remaining batch") as pbar:
        sequences = pipeline(
            prompts,
            max_new_tokens=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        for i, batch in enumerate(sequences):
            for seq in batch:
                generated_text = seq['generated_text']
                if "<Summary>:" in generated_text:
                    summarization = generated_text.split("<Summary>:")[1].strip()
                    summarization = summarization.split("\n\n", 1)[-1].strip()  # Extract part after "assistant\n\n"
                    df.at[rows_to_process[i], 'Summarization'] = summarization
                else:
                    df.at[rows_to_process[i], 'Summarization'] = generated_text.strip()
            pbar.update(1)

    # Append the remaining rows to the output CSV file
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for idx in rows_to_process:
            writer.writerow(df.loc[idx].values)

print(f"Data successfully processed and saved to {output_file}")
