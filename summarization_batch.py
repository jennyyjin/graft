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
# file_path = 'captions/combined_naip.csv'
file_path = 'captions/caption_sat_corres_6_cont.csv'
data = pd.read_csv(file_path)

# print(data.columns)

# Define a function to merge rows with the same SatelliteID and concatenate the captions
def merge_and_concatenate(df):
    # Group by 'Satellite Image Path'
    grouped = df.groupby('Satellite Image Path')

    # Function to concatenate captions and ground image paths
    def concatenate_data(group):
        # Remove line breaks from captions and concatenate them
        captions = [caption.replace('\n', ' ') for caption in group['Caption']]
        captions = "\n".join([f"{i+1}. {caption}" for i, caption in enumerate(captions)])

        # Concatenate ground image paths as comma-separated strings
        ground_image_ids = ", ".join(group['Ground Image Path'].astype(str))

        # Return the concatenated result as a series
        return pd.Series({
            'SatelliteID': group.name,  # use group name, which is 'Satellite Image Path'
            'Captions': captions,
            'GroundImageIDs': ground_image_ids
        })

    # Apply the function to each group and reset the index
    merged_data = grouped.apply(concatenate_data).reset_index(drop=True)
    
    # print(merged_data.columns)

    return merged_data

# Apply the function to the data
df = merge_and_concatenate(data)
# df.to_csv('captions/merged_caption_sat_corres_6_cont.csv')

# exit()

# Add a new column for summarization if it doesn't exist
if 'Summarization' not in df.columns:
    df['Summarization'] = ""

# print(df.columns)

# Output CSV file path
output_dir = 'summarizations/llama-3-8b-instruct'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'summarizations-llama-3-8b-instruct_batched_naip_6_cont_twoshot_ignore_small.csv')

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
        {"role": "user", "content": f"The texts in the brackets are descriptions \
            of images taken in the same region, ordered numerically without preference. \
            Summarize these descriptions to provide an overview of the region. \
            Avoid unnecessary preambles and transient objects, and also think from a higher level, \
            so ignore the ground level detail that cannot be seen from a satellite's point of view. \
            For example, if you see people and wildlives in the captions, do not include those directly, but infer to what this might possibly mean. \
            Here are some examples: \n\n     \
            Example1:\n[1. The image depicts a lush green field with a tree in \
            the foreground. The tree appears to be barren, with no leaves on \
            its branches. In the field, there are two cows, one closer to \
            the foreground and the other further back. The cows are grazing \
            on the grass, creating a serene and peaceful atmosphere in the scene. \
            2. The image features a lush green field with trees in the background. \
            The sky above the field is blue and cloudy, creating a serene atmosphere. \
            There are no people or animals visible in the scene, emphasizing the \
            natural beauty of the landscape.] \n \
            <Summary>: The region is characterized by a lush green field, surrounded \
            by trees and set under a blue, cloudy sky. It conveys a serene and \
            peaceful atmosphere. It might involve farms or golf court based on the area of the field. Overall, the \
            scenery showcases a blend of open green spaces with sporadic tree \
            cover, emphasizing its natural beauty. \n\n\
            Example2:\n[1. The image features a man in a yellow safety jacket working \
            on a crane. The crane is positioned above a business sign, possibly \
            a restaurant. The man appears to be fixing the crane, ensuring its \
            proper functioning.  In the scene, there is a truck parked near the \
            crane, and a bench can be seen on the right side of the image. Additionally, \
            there is a traffic light visible in the background, adding to the urban \
            setting of the scene.\n2. The image features a small restaurant situated \
            in a parking lot. The restaurant has a red roof and is surrounded by several \
            cars parked nearby. There is also a truck parked in the vicinity of the restaurant.  \
            In addition to the cars and the truck, there is a bench and a dining table visible \
            in the scene. The bench is placed close to the restaurant, while the dining table \
            is located further away, closer to the edge of the parking lot.\n3. The image \
            features a food truck parked on the side of a street. There are two people standing \
            in front of the food truck, likely waiting to be served. One person is positioned \
            closer to the left side of the truck, while the other person is standing a bit\
            further to the right.  The food truck appears to be selling Mexican food, \
            as evidenced by the presence of a bowl and a spoon in the scene. The bowl \
            is placed near the center of the truck, while the spoon is located closer \
            to the right side of the truck.\n4. The image features a train traveling \
            down the train tracks. The train is quite long, occupying a significant \
            portion of the scene. The tracks are located next to a body of water, \
            creating a picturesque backdrop for the train's journey. The sky above \
            the scene is cloudy, adding to the overall atmosphere of the image.\n5. \
            The image features a delicious meal consisting of a hamburger and a side \
            of french fries. The hamburger is placed in the center of the scene, while \
            the french fries are scattered around it. The hamburger appears to be a \
            cheeseburger, adding to the appetizing nature of the meal.] \n \
            <Summary>: The region in the image aligns with an urban area featuring \
            small commercial and food establishments. The presence of buildings with \
            flat roofs in a grid layout suggests businesses, potentially including a \
            small restaurant with a parking lot as described. The visible train tracks \
            at the bottom of the image match the earlier mention of tracks running \
            alongside a body of water. The parking spaces around some buildings \
            could correspond to locations where food trucks or outdoor dining \
            spaces, like benches and tables, might be set up, indicating a mix \
            of commercial and open areas for social activity. Overall, the region \
            combines urban commercial structures with transportation infrastructure,\
            matching the elements described in the initial captions. \n\n\
            Now, please summarize the following descriptions in the bracket: \n\n     \
            [{captions}] \n <Summary>:"}
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
                    # print(generated_text)
                    # print("\n a part in generated text by summary" + x for x in generated_text.split("<Summary>:"))
                    if "<Summary>:" in generated_text:
                        summarization = generated_text.split("<Summary>:")[-1].strip()
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
        # break

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
