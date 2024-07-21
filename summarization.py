import pandas as pd
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
import csv

# Load the model and tokenizer
model_dir = "../llama/llama-2-7b-chat-hf"
model = LlamaForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,  # Use mixed precision to save memory
    device_map="auto"  # Automatically split the model across available devices
)
tokenizer = LlamaTokenizer.from_pretrained(model_dir)

# Setup the pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Read the input CSV file
df = pd.read_csv('captions/sat_captions.csv')

# Add a new column for summarization if it doesn't exist
if 'Summarization' not in df.columns:
    df['Summarization'] = ""

# Output CSV file path
output_file = 'summarizations/summarizations-llama-2-7b-chat-hf.csv'

# Determine the last processed row
last_processed_index = -1
try:
    with open(output_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for last_processed_index, _ in enumerate(reader):
            pass
except FileNotFoundError:
    # Write the header to the output CSV file if it doesn't exist
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(df.columns)
    last_processed_index = 0  # Start from the beginning

# count = 0
# Process each row in the dataframe starting from the last processed row
for index, row in tqdm(df.iloc[last_processed_index + 1:].iterrows(), total=len(df) - last_processed_index - 1, desc="Processing rows"):
    captions = row['Captions']
  
    prompt = (
        f"The texts in the bracket are descriptions of images taken in the same region. Summarize the text to describe the region."
        f"[{captions}]"
        " <Summarization>:"
    )

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

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Optional: break after a certain number of rows for testing
    # count += 1
    # if count == 10:
    #     break
