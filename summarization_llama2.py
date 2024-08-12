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
    torch_dtype=torch.float16, 
    device_map="auto"  
)
tokenizer = LlamaTokenizer.from_pretrained(model_dir)

# Setup the pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Read the input CSV file
# df = pd.read_csv('captions/sat_captions_50k.csv')
# df = pd.read_csv('captions/sat_captions_100k.csv')
df = pd.read_csv('captions/sat_captions_100ks_recentered.csv')

# Add a new column for summarization if it doesn't exist
if 'Summarization' not in df.columns:
    df['Summarization'] = ""


# Output CSV file path
output_file = 'summarizations/summarizations-llama-2-7b-chat-hf_recentered.csv'
# print(df.columns)

# Determine the set of already processed SatelliteIDs
processed_satellite_ids = set()
try:
    with open(output_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            processed_satellite_ids.add(int(row[0]))  
except FileNotFoundError:
    # Write the header to the output CSV file if it doesn't exist
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(df.columns)
        
df = df[~df['SatelliteID'].isin(processed_satellite_ids)]  # Remoave processed satellite image
df = df[df['GroundImageIDs'].str.count(',') > 0] # Remove satellite images with less than 2 ground images

# count = 0
# Process each row in the dataframe
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    # satellite_id = row['SatelliteID']
  
    # # Skip if the satellite image has been processed
    # if satellite_id in processed_satellite_ids:
    #     continue

    captions = row['Captions']
    # captions = "1. The image features a small, square-shaped, brown ceramic box with a sunflower design on it. The box is placed on a wooden table, and the sunflower design is visible on the top of the box. The sunflower design is intricate and adds a decorative touch to the box. 2. The image features a beautiful beach scene with a clear blue sky above. The beach is situated next to a body of water, which appears to be a lake or a large pond. The water is calm and serene, creating a peaceful atmosphere. There are two wooden structures in the scene, one closer to the left side and the other towards the right side of the image. These structures seem to be part of a beachside area, possibly providing a place for people to relax and enjoy the view.There are two wooden structures in the scene, one closer to the left side and the other towards the right side of the image. These structures seem to be part of a beachside area, possibly providing a place for people to relax and enjoy the view."
    # captions = "Caption 1: The image features a small, square-shaped, brown ceramic box with a sunflower design on it. The box is placed on a wooden table, and it appears to be a decorative item or a small container for holding small items. The sunflower design adds a touch of color and artistic flair to the otherwise neutral-colored box. Caption 2: The image features a beach with a wooden pier extending out into the ocean. The pier is surrounded by a beautiful blue ocean, and there are clouds in the sky. The scene is captured through a glass door, providing a view of the beach and ocean from the comfort of a building."
    prompt = (
        f"The texts in the bracket are descriptions of images taken in the same region. Summarize the text to describe the region."
        f"[{captions}]"
        " <Summarization>:"
    )
    
    # print(prompt)

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
    # print(" ".join(text))
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
    # count += 1
    # if count == 10:
    #     break
    # break
