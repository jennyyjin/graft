import pandas as pd
import torch
import transformers
from transformers import LlamaForCausalLM, AutoTokenizer
from tqdm import tqdm
import csv
import os

# Hugging Face API token
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
df = df[~df['SatelliteID'].isin([4])]

# Add a new column for summarization if it doesn't exist
if 'Summarization' not in df.columns:
    df['Summarization'] = ""

# Output directory and file path
output_dir = 'summarizations/llama-3-8b-instruct'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'summarizations-llama-3-8b-instruct.csv')

# Initialize the output file with headers
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(df.columns)

# Get list of already processed SatelliteIDs
processed_satellite_ids = set()
if os.path.exists(output_file):
    existing_data = pd.read_csv(output_file)
    processed_satellite_ids = set(existing_data['SatelliteID'])

# Batch size
batch_size = 16

# example_prompts = (
#     "The texts in the bracket are descriptions of images taken in the same region. Summarize the text to describe the region, inferring the general location of the region.\n"
#     "Here are some examples:\n"
#     "[1. The image features a wooden dining table with a blue tray filled with a variety of food items. On the tray, there is a bowl of rice, a bowl of soup, a bowl of fish, and a bowl of noodles. In addition to the main dishes, there is a cup of tea and a cup of coffee on the table. A cell phone can be seen on the left side of the table, possibly belonging to one of the people enjoying the meal. The dining table occupies most of the space in the image, showcasing the assortment of dishes and the inviting atmosphere of the meal.]\n"
#     "<Summarization>: The image features a wooden dining table with a blue tray containing bowls of rice, soup, fish, and noodles, along with a cup of tea, a cup of coffee, and a cell phone. Given the setting with a variety of dishes and personal items like a cell phone, the scene likely depicts a residential area, suggesting a home dining setting where a meal is being enjoyed. \n\n"
#     "[1. The image depicts a large white building with many windows, likely an office building. The building is situated next to a lush green tree, which adds a touch of nature to the scene. The tree is positioned on the left side of the building, covering a significant portion of the image. In addition to the building and the tree, there are several cars parked in front of the building. Some cars are parked closer to the left side of the building, while others are positioned more towards the right side. The combination of the building, tree, and parked cars creates an urban setting with a touch of greenery. 2. The image features a large, white building with many windows. The building appears to be a hotel or an apartment complex, as it has many balconies on the upper floors. The building is situated next to a tree, which adds a touch of greenery to the scene. In front of the building, there is a lamp post, providing illumination to the area. The lamp post is positioned near the center of the image, and it appears to be the main source of light in the scene. The combination of the white building, the tree, and the lamp post creates a pleasant and inviting atmosphere. 3. The image depicts a large library filled with numerous bookshelves. The shelves are stocked with a variety of books in different sizes and colors. The books are neatly arranged on the shelves, creating an organized and visually appealing display. In addition to the bookshelves, there are several chairs placed throughout the library, likely for patrons to sit and read or study. The library appears to be well-maintained and inviting, providing a comfortable environment for visitors to explore and enjoy the vast collection of books.]\n"
#     "<Summarization>: The images collectively depict various aspects of an urban area with mixed-use buildings and public amenities. One image shows a large white office building with many windows and a nearby green tree, cars parked in front, and an urban setting with a touch of nature. Another image features a similar white building, possibly a hotel or apartment complex with balconies, a tree, and a lamp post, suggesting a pleasant and inviting atmosphere. The final image depicts an interior scene of a large library with well-organized bookshelves and seating, indicating a public space designed for reading and studying. The combination of these scenes suggests the region is a well-maintained urban area with residential, commercial, and public spaces.\n\n"
# )

# Function to process a batch of prompts
# def process_batch(batch):
    # Prepare batch with examples and actual data
    # batch_with_examples = [example_prompts + f"[{captions}]\n<Summarization>:" for captions in batch]
    
    
    # print(batch_with_examples)
    
    # # Generate the summarizations
    # sequences = pipeline(
    #     batch_with_examples,
    #     do_sample=True,
    #     top_k=10,
    #     num_return_sequences=1,
    #     max_new_tokens=200,
    # )

    # # Extract the generated text
    # results = []
    # for seq in sequences:
    #     for s in seq:
    #         generated_text = s['generated_text']
    #         if "<Summarization>:" in generated_text:
    #             summarization = generated_text.split("<Summarization>:")[1].strip()
    #             results.append(summarization)
    #         else:
    #             results.append(generated_text.strip())
    # return results
    
def process_batch(batch):
    # Generate the summarizations
    sequences = pipeline(
        batch,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        max_new_tokens=200,
    )

    # Extract the generated text
    results = []
    for seq in sequences:
        for s in seq:
            generated_text = s['generated_text']
            if "<Summarization>:" in generated_text:
                summarization = generated_text.split("<Summarization>:")[1].strip()
                results.append(summarization)
            else:
                results.append(generated_text.strip())
    return results

# Prepare the batches
batches = []
current_batch = []
current_indices = []
for index, row in tqdm(df.iterrows(), total=len(df), desc="Preparing batches"):
    satellite_id = row['SatelliteID']
    
    # Skip if the satellite image has been processed
    if satellite_id in processed_satellite_ids:
        continue

    captions = row['Captions']
    # prompt = (
    #     f"The texts in the bracket are descriptions of images taken in the same region. Summarize the text to describe the region, inferring the general location of the region.\n"
    #     "Here are some examples:\n"
    #     "[1. The image features a wooden dining table with a blue tray filled with a variety of food items. On the tray, there is a bowl of rice, a bowl of soup, a bowl of fish, and a bowl of noodles. In addition to the main dishes, there is a cup of tea and a cup of coffee on the table.   A cell phone can be seen on the left side of the table, possibly belonging to one of the people enjoying the meal. The dining table occupies most of the space in the image, showcasing the assortment of dishes and the inviting atmosphere of the meal.]\n"
    #     "<Summarization>: The image features a wooden dining table with a blue tray containing bowls of rice, soup, fish, and noodles, along with a cup of tea, a cup of coffee, and a cell phone. Given the setting with a variety of dishes and personal items like a cell phone, the scene likely depicts a residential area, suggesting a home dining setting where a meal is being enjoyed. \n\n"
    #     "[1. The image depicts a large white building with many windows, likely an office building. The building is situated next to a lush green tree, which adds a touch of nature to the scene. The tree is positioned on the left side of the building, covering a significant portion of the image.   In addition to the building and the tree, there are several cars parked in front of the building. Some cars are parked closer to the left side of the building, while others are positioned more towards the right side. The combination of the building, tree, and parked cars creates an urban setting with a touch of greenery. 2. The image features a large, white building with many windows. The building appears to be a hotel or an apartment complex, as it has many balconies on the upper floors. The building is situated next to a tree, which adds a touch of greenery to the scene.   In front of the building, there is a lamp post, providing illumination to the area. The lamp post is positioned near the center of the image, and it appears to be the main source of light in the scene. The combination of the white building, the tree, and the lamp post creates a pleasant and inviting atmosphere. 3. The image depicts a large library filled with numerous bookshelves. The shelves are stocked with a variety of books in different sizes and colors. The books are neatly arranged on the shelves, creating an organized and visually appealing display.   In addition to the bookshelves, there are several chairs placed throughout the library, likely for patrons to sit and read or study. The library appears to be well-maintained and inviting, providing a comfortable environment for visitors to explore and enjoy the vast collection of books. ]\n"
    #     "<Summarization>: The images collectively depict various aspects of an urban area with mixed-use buildings and public amenities. One image shows a large white office building with many windows and a nearby green tree, cars parked in front, and an urban setting with a touch of nature. Another image features a similar white building, possibly a hotel or apartment complex with balconies, a tree, and a lamp post, suggesting a pleasant and inviting atmosphere. The final image depicts an interior scene of a large library with well-organized bookshelves and seating, indicating a public space designed for reading and studying. The combination of these scenes suggests the region is a well-maintained urban area with residential, commercial, and public spaces.\n\n"
    #     f"[{captions}]\n"
    #     "<Summarization>:"
    # )
    # print(prompt)
    prompt = (
        f"The texts in the bracket are descriptions of images taken in the same region. Summarize the text to describe the region. [{captions}] <Summarization>:"
    )
    # print(prompt)
    current_batch.append(captions)
    current_indices.append(index)
    
    if len(current_batch) == batch_size:
        batches.append((current_batch, current_indices))
        current_batch = []
        current_indices = []
    # break

# exit()

# Process each batch
for batch, indices in tqdm(batches, desc="Processing batches"):
    summaries = process_batch(batch)
    for summary, idx in zip(summaries, indices):
        df.at[idx, 'Summarization'] = summary
        processed_satellite_ids.add(df.at[idx, 'SatelliteID'])

    # Append the updated rows to the output CSV file
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for idx in indices:
            writer.writerow(df.loc[idx].values)

    # Clear GPU cache
    torch.cuda.empty_cache()
    break

# Process any remaining prompts in the last batch
if current_batch:
    summaries = process_batch(current_batch)
    for summary, idx in zip(summaries, current_indices):
        df.at[idx, 'Summarization'] = summary
        processed_satellite_ids.add(df.at[idx, 'SatelliteID'])

    # Append the updated rows to the output CSV file
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for idx in current_indices:
            writer.writerow(df.loc[idx].values)

# Clear GPU cache one last time
torch.cuda.empty_cache()
