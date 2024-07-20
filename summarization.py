import pandas as pd
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
import csv

# Load the model and tokenizer
model_dir = "../llama/llama-2-7b-chat-hf"
model = LlamaForCausalLM.from_pretrained(model_dir)
tokenizer = LlamaTokenizer.from_pretrained(model_dir)

# Setup the pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    device="cuda:0",
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Read the input CSV file
df = pd.read_csv('captions/sat_captions.csv')

# Add a new column for summarization if it doesn't exist
if 'Summarization' not in df.columns:
    df['Summarization'] = ""

# Output CSV file path
output_file = 'summarizations/summarizations-llama-2-7b-chat-hf.csv'

# Write the header to the output CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(df.columns)

# Process each row in the dataframe
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
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
        max_length=600,
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
