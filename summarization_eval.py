import pandas as pd
import torch
import transformers
from transformers import LlamaForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
from datasets import Dataset, DatasetDict
import csv

# Hugging Face API token
hf_token = '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'

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model = LlamaForCausalLM.from_pretrained(
    model_name,
    token=hf_token,  # Use the provided Hugging Face token
    torch_dtype=torch.float16,  # Use mixed precision to save memory
    device_map="auto"  # Automatically split the model across available devices
)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

# Setup the pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# List of input CSV files
input_files = [f'summarizations-{i}-llama-2-7b-chat-hf.csv' for i in range(1, 5)]
input_files = [os.path.join('summarizations/llama-2-7b-chat-hf', file) for file in input_files]

# Load the CSV files into a single dataset
datasets = [pd.read_csv(file) for file in input_files]

# Merge the datasets on SatelliteID, Captions, and GroundImageIDs
merged_df = pd.concat(datasets, keys=[f'summarization_{i}' for i in range(1, 5)])
merged_df = merged_df.reset_index(level=0).rename(columns={'level_0': 'Source'})
merged_df['Source'] = merged_df['Source'].apply(lambda x: int(x.split('_')[1]))

# Group by unique identifiers
grouped = merged_df.groupby(['SatelliteID', 'Captions', 'GroundImageIDs'])

# Create a new dataframe to hold the rankings
rankings_df = pd.DataFrame(columns=['SatelliteID', 'Captions', 'GroundImageIDs', 'Rankings'])

# Ranking prompt
ranking_prompt = (
    "You are given several summaries of the same region. Rank these summaries based on their quality in summarizing the necessary information and inferring the general location of the region. Summaries: [{}] <Ranking>:"
)

# Function to rank summarizations
def rank_summarizations(group):
    summaries = group['Summarization'].tolist()
    sources = group['Source'].tolist()
    prompt = ranking_prompt.format(" | ".join(summaries))
    print(prompt)

    # Generate the ranking
    sequences = pipeline(
        prompt,
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=50,
    )

    # Extract the generated text
    ranking = sequences[0]['generated_text'].split("<Ranking>:")[1].strip()
    # ranked_order = ranking.split(',')

    return {
        'SatelliteID': group['SatelliteID'].iloc[0],
        'Captions': group['Captions'].iloc[0],
        'GroundImageIDs': group['GroundImageIDs'].iloc[0],
        'Rankings': ranked_order
    }

# Save the ranked summarizations in real-time
ranked_output_file = os.path.join('summarizations/llama-2-7b-chat-hf', 'ranked_summarizations.csv')

# Initialize the CSV file with the header
with open(ranked_output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['SatelliteID', 'Captions', 'GroundImageIDs', 'Rankings'])

# Process and rank each group
for name, group in tqdm(grouped, desc="Processing groups"):
    ranked_data = rank_summarizations(group)
    
    # Append the result to the CSV file
    with open(ranked_output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([ranked_data['SatelliteID'], ranked_data['Captions'], ranked_data['GroundImageIDs'], ranked_data['Rankings']])
