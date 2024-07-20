import multiprocessing
multiprocessing.set_start_method('spawn', force=True)  # Set the start method at the top

import pandas as pd
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
import csv

model_dir = "../llama/llama-2-7b-chat-hf"
model = LlamaForCausalLM.from_pretrained(model_dir)

tokenizer = LlamaTokenizer.from_pretrained(model_dir)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    device="cuda:0",
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

df = pd.read_csv('captions/sat_captions.csv')

if 'Summarization' not in df.columns:
    df['Summarization'] = ""

output_file = 'summarizations/summarizations-llama-2-7b-chat-hf.csv'

# Write header to output file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(df.columns)

def generate_summarization(row):
    index, row_data = row
    captions = row_data['Captions']

    prompt = (
        f"The texts in the bracket are descriptions of images taken in the same region. Summarize the text to describe the region."
        f"[{captions}]"
        " <Summarization>:"
    )

    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=600,
    )

    text = []
    for seq in sequences:
        generated_text = seq['generated_text']
        if "<Summarization>:" in generated_text:
            summarization = generated_text.split("<Summarization>:")[1].strip()
            text.append(summarization)
        else:
            text.append(generated_text.strip())

    summarization = " ".join(text)
    row_data['Summarization'] = summarization
    return row_data

def process_row(row):
    result = generate_summarization(row)
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result.values)
    return result

if __name__ == '__main__':
    rows = list(df.iterrows())

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        with tqdm(total=len(rows), desc="Processing rows") as pbar:
            for _ in pool.imap_unordered(process_row, rows):
                pbar.update()
