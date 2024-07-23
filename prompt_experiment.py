import pandas as pd
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
import csv
import os

# no need to choose the prompt now, can experiment later
# try out llama instruct
# generate more summarizations
# do a one shot or few shot prompting

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

# List of prompt formats
prompts = [
    "The texts in the bracket are descriptions of images taken in the same region. Summarize the text to describe the region. [{}] <Summarization>:",
    "Summarize the following image descriptions to give an overview of the region: [{}] <Summarization>:",
    "Provide a summary for the region based on the following descriptions: [{}] <Summarization>:", 
    "You are a satellite image researcher. Generate the description of the region based on the images. You can make reasonable correlations if necessary. [{}] <Summarization>:" 
]

# Output directory
output_dir = 'summarizations/llama-2-7b-chat-hf'
os.makedirs(output_dir, exist_ok=True)

log_file_path = os.path.join(output_dir, 'prompt_log.txt')
with open(log_file_path, 'w') as log_file:
    log_file.write('')

# Process the first 1000 rows for each prompt
for prompt_format in prompts:
    output_file = os.path.join(output_dir, f'summarizations-{prompts.index(prompt_format) + 1}-llama-2-7b-chat-hf.csv')
    print("Starting summarization for prompt format:", prompt_format)
    # Write the header to the output CSV file if it doesn't exist
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(df.columns)
        
    first_prompt_logged = False

    # Process each row in the dataframe
    for index, row in tqdm(df.iloc[:2000].iterrows(), total=2000, desc=f"Processing rows for prompt {prompts.index(prompt_format) + 1}"):
        captions = row['Captions']
        prompt = prompt_format.format(captions)
        print(prompt)
        
        # Log the first prompt
        if not first_prompt_logged:
            with open(log_file_path, 'a') as log_file:
                log_file.write(f'First prompt for prompt format {prompts.index(prompt_format) + 1}:\n{prompt}\n\n')
            first_prompt_logged = True

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
