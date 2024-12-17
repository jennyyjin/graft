import pandas as pd
import os
import csv
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
import pickle
from transformers import AutoTokenizer

# Load VLLM LLaMA 3 8B model
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# hf_token = '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'

# Initialize VLLM LLM model with parallelism
llm = LLM(model=model_name)

# 1: 6_cont
# 2: 0_250
# 3: 251_475

# Load the CSV file
csv_name = '6_cont'
csv_order = '1'
file_path = f'captions/caption_sat_corres_{csv_name}.csv'
sat_centers_path = '/share/kavita/ukm4/datasets/CLIPRS10m_m2o/src/satellite_centers.pkl'
ground_coords_path = f'ground_image_coordinates_{csv_order}.csv'
data = pd.read_csv(file_path)
ground_coords = pd.read_csv(ground_coords_path)

def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

sat_centers = load_from_pickle(sat_centers_path)

def get_region(ground_coord, satellite_center, halfwidth):
    """
    Determine which part of the satellite region the ground coordinate falls into
    based on the satellite center and the halfwidth.
    
    The region is divided into nine parts:
    north, south, east, west, center, northeast, northwest, southeast, southwest.
    
    Arguments:
    - ground_coord: tuple of (latitude, longitude) for the ground image.
    - satellite_center: tuple of (latitude, longitude) for the satellite center.
    - halfwidth: the distance from the satellite center to the edge in latitude and longitude.
    
    
    TODO: Adjust the halfwidth value according to your dataset.
    
    Returns:
    - A string describing which part of the satellite region the ground image is in.
    """
    lat, lon = ground_coord
    center_lat, center_lon = satellite_center
    
    # Calculate boundaries for each part
    north_boundary = center_lat + halfwidth/6
    south_boundary = center_lat - halfwidth/6
    east_boundary = center_lon + halfwidth/6
    west_boundary = center_lon - halfwidth/6
    
    if lat > north_boundary:
        if lon > east_boundary:
            return 'northeast'
        elif lon < west_boundary:
            return 'northwest'
        else:
            return 'north'
    elif lat < south_boundary:
        if lon > east_boundary:
            return 'southeast'
        elif lon < west_boundary:
            return 'southwest'
        else:
            return 'south'
    else:
        if lon > east_boundary:
            return 'east'
        elif lon < west_boundary:
            return 'west'
        else:
            return 'center'
        
def concatenate_data_with_region(group):
    captions = []
    for i, row in group.iterrows():
        # Get the ground coordinate for the current row
        # print(row['coords'])
        # print(row['coords'][1:-1].split(','))
        # check whether row['coords'] is nan
        if not pd.isna(row['coords']):
            ground_coord = tuple(map(float, row['coords'][1:-1].split(',')))  # Convert from string to tuple
            satellite_center_coord = sat_centers['PoIs'][int(group.name.split('/')[-1].split('.')[0])]
            
            # Define the halfwidth (adjust this value as needed)
            halfwidth = 0.0021  # Example halfwidth value, adjust according to your dataset
            
            # Determine which part of the satellite region this image belongs to
            region = get_region(ground_coord, satellite_center_coord, halfwidth)
            
            # Prepend the region description to the caption
            caption_with_region = f"This image is taken in the {region} of the satellite region. " + row['Caption'].replace('\n', ' ')
        else:
            caption_with_region = row['Caption'].replace('\n', ' ')
        captions.append(f"{i+1}. {caption_with_region}")
    
    # Join all captions
    captions = "\n".join(captions)

    # Concatenate ground image paths and coordinates
    ground_image_ids = ", ".join(group['Ground Image Path'].astype(str))
    coords = ", ".join(group['coords'].astype(str))
    
    satellite_center_coord = sat_centers['PoIs'][int(group.name.split('/')[-1].split('.')[0])]

    return pd.Series({
        'SatelliteID': group.name,
        'Captions': captions,
        'GroundImageIDs': ground_image_ids,
        'GroundCoordinates': coords,
        'SatelliteCenterCoord': satellite_center_coord
    })


# Define a function to merge rows with the same SatelliteID and concatenate the captions
def merge_and_concatenate(df, ground_coords):
    # Create 'coords' column in ground_coords combining 'latitude' and 'longitude'
    ground_coords['coords'] = ground_coords.apply(lambda row: f"({row['latitude']}, {row['longitude']})", axis=1)

    # Extract ground_image_id from the Ground Image Path in df and convert to string
    df['ground_image_id'] = df['Ground Image Path'].apply(lambda x: x.split('/')[-1].split('.')[0])
    ground_coords['ground_image_id'] = ground_coords['ground_image_id'].astype(str)

    # Merge df with ground_coords based on ground_image_id
    df = df.merge(ground_coords[['ground_image_id', 'coords']], on='ground_image_id', how='left')

    # Group by 'Satellite Image Path'
    grouped = df.groupby('Satellite Image Path')

    # Apply the concatenate_data_with_region function to each group
    merged_data = grouped.apply(concatenate_data_with_region).reset_index(drop=True)
    return merged_data

# Apply the function to the data
df = merge_and_concatenate(data, ground_coords)
df.to_csv(f'merged_data_{csv_order}.csv')

# Add a new column for summarization if it doesn't exist
if 'Summarization' not in df.columns:
    df['Summarization'] = ""

# Output CSV file path
output_dir = 'summarizations/llama-3-8b-instruct'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f'summarizations-llama-3-8b-instruct_batched_naip_{csv_name}_grid.csv')

# Load previously processed data, if available
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
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(df.columns)
    filtered_df = df
    
def construct_prompt(messages):
    prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        if role == "system":
            # System messages can be prepended without any role indicator
            prompt += f"{content}\n\n"
        elif role == "user":
            # User messages can be prefixed with "User:"
            prompt += f"{content}\n\n"
        elif role == "assistant":
            # Assistant messages can be prefixed with "Assistant:"
            prompt += f"{content}\n\n"
    return prompt


# Set up VLLM sampling parameters
sampling_params = SamplingParams(
    max_tokens=250,
    temperature=0.7,
    top_k=10,
    top_p=0.9,
)

# prompt_template = """
#     You are a helpful AI assistant for summarizing image descriptions from the same region taken by a satellite.

#     The texts within the brackets are descriptions of images taken in the same region, numbered sequentially without preference. Your task is to combine these descriptions to provide an accurate, high-level overview of the region based on satellite imagery. **Do not include or explicitly mention any transient objects or small-scale details that would not be visible from a satellite’s perspective.**

#     Guidelines:

#     1. **Aerial Perspective:**
#     - Summarize from a satellite's point of view.
#     - **Ignore any ground-level details** such as:
#         - Individual people, animals (like dogs, cows, birds, or wildlife), small vehicles (bicycles), or other small objects.
#         - Small architectural details (e.g., windows, decorations) and any specific action happening on the ground level.

#     2. **Infer Rather Than Describe:**
#     - If people, animals, or transient objects appear in the captions, **do not include them directly**. Instead, infer what their presence might indicate about the region:
#         - **People**: Suggest human activities or an urban/commercial area.
#         - **Cows**: Indicate farmland or agricultural land.
#         - **Birds/Wildlife**: Ignore unless they provide clues about the environment (e.g., wildlife could hint at a natural habitat).
#         - **Signs/Billboards**: Ignore unless they suggest a specific type of area (e.g., commercial or industrial), if they suggest a specific type of area, also do not mention the object itself, just the property of the area.

#     3. **Focus on Permanent, Physical Attributes:**
#     - Describe only large-scale and permanent features that are visible from above (e.g., fields, buildings, bodies of water, transportation infrastructure like roads and railways).
#     - Avoid any mention of atmospheric conditions or transient qualities like a "peaceful" or "busy" atmosphere.
#     - For example, if a caption mentions a "crowded area with many people," summarize as "an urban or commercial area."

#     4. **Emphasize Consistency Across Images:**
#     - Summarize as if you are describing a single cohesive region. Focus on recurring features and ignore any small, non-permanent elements that vary from image to image.

#     5. **Use Bullet Points:**
#     - Present the summarized details in bullet points for clarity and conciseness.

#     Examples:

#     ---

#     **Correct Example 1:**

#     Please summarize the following descriptions within the brackets:
    
#     Captions:

#     [
#     1. This image is taken in the north of the satellite region. The image depicts a lush green field with a tree in the foreground. The tree appears barren, with no leaves on its branches. In the field, there are two cows grazing on the grass.

#     2. This image is taken in the northwest of the satellite region. The image features a lush green field with trees in the background. The sky above is blue and cloudy. There are no people or animals visible, emphasizing the natural beauty of the landscape.
#     ]

#     Summary:

#     - The region, primarily in the northern part, is characterized by a lush green field.
#     - Trees are present in the background, particularly in the northwest, indicating possible farmland.
#     - Large open spaces suggest agricultural or rural land use.

#     ---

#     **Correct Example 2:**
    
#     Please summarize the following descriptions within the brackets:

#     Captions:

#     [
#     1. This image is taken in the east of the satellite region. The image shows a crowded street with people walking and cars parked on the side. There are shops and small restaurants visible along the street.

#     2. This image is taken in the southeast of the satellite region. The image features a food truck parked beside a building, with people standing in line. Some small plants are visible near the truck.
#     ]
#     Summary:

#     - The region, primarily in the eastern and southeastern parts, is an urban area.
#     - Small commercial establishments like shops and restaurants are visible in both sections.
#     - The presence of transportation infrastructure such as streets and parked cars is evident throughout.

#     ---

#     **Incorrect Example 1:**
#     Please summarize the following descriptions within the brackets:

#     Captions:

#     [
#     1. This image is taken in the center of the satellite region. The image features a large brick tower with a red and black color scheme. The tower stands prominently in the foreground, towering over the surrounding landscape. The sky serves as a backdrop for the impressive structure, enhancing its presence in the scene.

#     2. This image is taken in the south of the satellite region. The image features a large red and black lighthouse towering over the surrounding area. The lighthouse is prominently visible in the scene, with its vibrant colors standing out against the backdrop. The sky above the lighthouse appears to be clear and blue, adding to the serene atmosphere of the scene.
#     ]
#     Summary:

#     - The region is characterized by large brick towers, possibly lighthouses, located in the center and southern parts.
#     - Some areas feature grassy spaces, houses, cars, and clock towers, suggesting a mix of historical and residential areas.
#     - Stained glass windows and other small architectural details are visible.
#     - The overall region includes a combination of urban and natural environments.

#     **Reason for Incorrectness:**

#     - Should not mention atmosphere or ambience (such as the sky or serene atmosphere).
#     - Describing multiple scenes instead of summarizing the region as a whole.
#     - References to small architectural details like stained glass windows are not visible from a satellite view.
    
#     Corrected Summary:

#     - The region is characterized by large brick towers, possibly lighthouses, visible in both the central and southern parts of the satellite region.
#     - The towers are prominent landmarks and stand out in the surrounding landscape.
#     - The surrounding area includes open land with few significant structures.
#     ---

#     **Incorrect Example 2:**
#     Please summarize the following descriptions within the brackets:

#     Captions:

#     [
#     1. This image is taken in the northwest of the satellite region. The image features a train traveling down the tracks. The train is quite large and occupies a significant portion of the scene. The train appears to be passing through a countryside area, with trees visible in the background. Additionally, there is a crane in the foreground of the image, adding an interesting element to the scene.
#     ]

#     Summary:

#     - The region features train tracks running through a countryside area in the northwest.
#     - The presence of a crane in the foreground suggests industrial or construction activity.
#     - Trees in the background suggest a rural environment.

#     **Reason for Incorrectness:**

#     - Should not mention transient objects like the crane directly.
#     - The summary should focus on larger-scale features like transportation infrastructure (the train track) and the countryside.

#     Corrected Summary:

#     - The region in the northwest features train tracks running through a countryside area.
#     - Possible industrial or construction activities inferred from nearby infrastructure.
#     - Trees in the background suggest a rural environment.

#     ---

#     Your Task:

#     Please summarize the following descriptions within the brackets following the previous exmaples and guidelines:
    
#     Captions:
#     [{captions}]

#     **Summary:**
#     """

# prompt_template = """
# You are a helpful AI assistant for summarizing image descriptions from the same region taken by a satellite.
# The texts within the brackets are descriptions of images taken in the same region. Your task is to combine these descriptions to provide an accurate, high-level overview of the region based on satellite imagery.

# **Guidelines:**
# 1. **Aerial Perspective:**
#    - Summarize from a satellite's point of view.
#    - **Ignore ground-level details** such as individual people, animals, small vehicles, small architectural details, and specific ground-level actions.
# 2. **Infer Rather Than Describe:**
#    - If transient objects appear, infer what their presence might indicate about the region without mentioning them directly.
# 3. **Focus on Permanent, Physical Attributes:**
#    - Describe only large-scale and permanent features visible from above.
#    - Avoid mentioning atmospheric conditions or transient qualities like "peaceful" or "busy" atmosphere.
# 4. **Emphasize Consistency Across Images:**
#    - Summarize as if describing a single cohesive region, focusing on recurring features.
# 5. **Use Bullet Points:**
#    - Present the summarized details in bullet points for clarity and conciseness.

# **Here's an Example:**
# Captions:
# [
# 1. This image is taken in the east of the satellite region. The image shows a crowded street with people walking and cars parked on the side. There are shops and small restaurants visible along the street.

# 2. This image is taken in the southeast of the satellite region. The image features a food truck parked beside a building, with people standing in line. Some small plants are visible near the truck.
# ]
# Summary:
# - The region in the eastern and southeastern parts is an urban area.
# - Presence of commercial establishments like shops and restaurants.
# - Visible transportation infrastructure such as streets and parked cars.

# **Now, please provide a summary of the following captions according to the guidelines. Remember to output only the summary in bullet points and nothing else.**
# Captions:
# [{captions}]
# Summary:
# """




batch_size = 16  # Define batch size
prompts = []
rows_to_process = []

# Create prompts and batches
for index, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Processing rows"):
    captions = row['Captions']
    # Define the entire prompt as a formatted string
    messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant for summarizing image descriptions from the same region taken by a satellite."
    },
    {
        "role": "user",
        "content": f"""
    The texts within the brackets are descriptions of images taken in the same region, numbered sequentially without preference. Your task is to combine these descriptions to provide an accurate, high-level overview of the region based on satellite imagery. **Do not include or explicitly mention any transient objects or small-scale details that would not be visible from a satellite’s perspective.**

    Guidelines:

    1. **Aerial Perspective:**
    - Summarize from a satellite's point of view.
    - **Ignore any ground-level details** such as:
        - Individual people, animals (like dogs, cows, birds, or wildlife), small vehicles (bicycles), or other small objects.
        - Small architectural details (e.g., windows, decorations) and any specific action happening on the ground level.

    2. **Infer Rather Than Describe:**
    - If people, animals, or transient objects appear in the captions, **do not include them directly**. Instead, infer what their presence might indicate about the region:
        - **People**: Suggest human activities or an urban/commercial area.
        - **Cows**: Indicate farmland or agricultural land.
        - **Birds/Wildlife**: Ignore unless they provide clues about the environment (e.g., wildlife could hint at a natural habitat).

    3. **Focus on Permanent, Physical Attributes:**
    - Describe only large-scale and permanent features that are visible from above (e.g., fields, buildings, bodies of water, transportation infrastructure like roads and railways).
    - Avoid any mention of atmospheric conditions or transient qualities like a "peaceful" or "busy" atmosphere.
    - For example, if a caption mentions a "crowded area with many people," summarize as "an urban or commercial area."

    4. **Emphasize Consistency Across Images:**
    - Summarize as if you are describing a single cohesive region. Focus on recurring features and ignore any small, non-permanent elements that vary from image to image.

    5. **Use Bullet Points:**
    - Present the summarized details in bullet points for clarity and conciseness.

    Examples:

    ---

    **Correct Example 1:**

    Captions:

    [
    1. The image depicts a lush green field with a tree in the foreground. The tree appears barren, with no leaves on its branches. In the field, there are two cows grazing on the grass.

    2. The image features a lush green field with trees in the background. The sky above is blue and cloudy. There are no people or animals visible, emphasizing the natural beauty of the landscape.
    ]

    **Summary (Bullet Points):**

    - The region is characterized by a lush green field.
    - Trees are present in the background, indicating possible farmland.
    - Large open spaces suggest agricultural or rural land use.

    ---

    **Correct Example 2:**

    Captions:

    [
    1. The image shows a crowded street with people walking and cars parked on the side. There are shops and small restaurants visible along the street.

    2. The image features a food truck parked beside a building, with people standing in line. Some small plants are visible near the truck.
    ]

    **Summary (Bullet Points):**

    - The region is an urban area.
    - Small commercial establishments like shops and restaurants are visible.
    - The presence of transportation infrastructure such as streets and parked cars.

    ---

    **Incorrect Example 1:**

    Captions:

    [
    1. The image features a large brick tower with a red and black color scheme. The tower stands prominently in the foreground, towering over the surrounding landscape. The sky serves as a backdrop for the impressive structure, enhancing its presence in the scene.

    2. The image features a large red and black lighthouse towering over the surrounding area. The lighthouse is prominently visible in the scene, with its vibrant colors standing out against the backdrop. The sky above the lighthouse appears to be clear and blue, adding to the serene atmosphere of the scene.
    ]

    **Incorrect Summary (Bullet Points):**

    - The region is characterized by large brick towers, possibly lighthouses.
    - Some areas feature grassy spaces, houses, cars, and clock towers, suggesting a mix of historical and residential areas.
    - Stained glass windows and other small architectural details are visible.
    - The overall region includes a combination of urban and natural environments.

    **Reason for Incorrectness:**

    - Should not mention atmosphere or ambience (such as the sky or serene atmosphere).
    - Describing multiple scenes instead of summarizing the region as a whole.
    - References to small architectural details like stained glass windows are not visible from a satellite view.

    ---

    **Incorrect Example 2:**

    Captions:

    [
    1. The image features a train traveling down the tracks. The train is quite large and occupies a significant portion of the scene. The train appears to be passing through a countryside area, with trees visible in the background. Additionally, there is a crane in the foreground of the image, adding an interesting element to the scene.
    ]

    **Incorrect Summary (Bullet Points):**

    - The region features train tracks running through a countryside area.
    - The presence of a crane in the foreground suggests industrial or construction activity.
    - Trees in the background suggest a rural environment.

    **Reason for Incorrectness:**

    - Should not mention transient objects like the crane directly.
    - The summary should focus on larger-scale features like transportation infrastructure (the train track) and the countryside.

    **Corrected Summary (Bullet Points):**

    - The region features train tracks running through a countryside area.
    - Possible industrial or construction activities inferred from nearby infrastructure.
    - Trees in the background suggest a rural environment.

    ---

    Your Task:

    Now, please summarize the following descriptions within the brackets:

    [{captions}]

    **Summary (Bullet Points):**
    """
        }
    ]
    
    # prompt = prompt_template.format(captions=row['Captions'])
    # prompt = construct_prompt(messages)
    prompt =  tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # print(index)
    # print(row['Captions'])
    prompts.append(prompt)
    rows_to_process.append(index)
    
    # Process batch
    if len(prompts) == batch_size:
        with tqdm(total=len(prompts), desc="Processing batch") as pbar:
            # Generate outputs using VLLM
            responses = llm.generate(prompts, sampling_params)
            
            for i, response in enumerate(responses):
                generated_text = response.outputs[0].text
                lines = generated_text.split('\n')
                bullet_points = [line for line in lines if line.strip().startswith('•')]
                bullet_text = '\n'.join(bullet_points)
                df.at[rows_to_process[i], 'Summarization'] = bullet_text
                # if "Summary:" in generated_text:
                #     summarization = generated_text.split("**Summary (Bullet Points):**")[-1].strip()
                #     df.at[rows_to_process[i], 'Summarization'] = summarization
                # else:
                #     df.at[rows_to_process[i], 'Summarization'] = generated_text.strip()
                pbar.update(1)
                
        # break

        # Append the updated rows to the output CSV file
        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for idx in rows_to_process:
                writer.writerow(df.loc[idx].values)
        
        prompts.clear()
        rows_to_process.clear()
        # break

# Process any remaining rows
if prompts:
    with tqdm(total=len(prompts), desc="Processing remaining batch") as pbar:
        responses = llm.generate(prompts, sampling_params)
        
        for i, response in enumerate(responses):
            generated_text = response.outputs[0].text
            lines = generated_text.split('\n')
            bullet_points = [line for line in lines if line.strip().startswith('•')]
            bullet_text = '\n'.join(bullet_points)
            df.at[rows_to_process[i], 'Summarization'] = bullet_text

            pbar.update(1)

    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for idx in rows_to_process:
            writer.writerow(df.loc[idx].values)

print(f"Data successfully processed and saved to {output_file}")
