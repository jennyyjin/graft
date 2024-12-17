import openai
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import csv
import os
import time
import threading
from openai import OpenAIError, RateLimitError

client = openai.OpenAI()  # Assuming this is how you set up OpenAI client, check for exact syntax

# Lock to ensure CSV writing is thread-safe
csv_lock = threading.Lock()

# Store results and errors in memory for batch processing and logging
results = []
error_log = []

def generate_instructions(image_caption, coordinates, location, labels, retries=5):
    system_prompt = """
    You are an AI visual assistant that can analyze a single image. You receive several sentences, each describing the same satellite image you are observing. In addition, specific object locations within the image are given, along with detailed coordinates. One label of the object might have multiple bounding boxes, resulting in a list of bounding boxes contained by '[]'. These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
    
    Your goal is to create questions and answers that accurately refer to specific areas, objects, or regions using precise spatial language and geographical cues. For each image, utilize the given bounding boxes and coordinates to pinpoint exact regions, ensuring that the instructions make it easy for someone unfamiliar with the image to locate the objects based on the provided information. The questions and answers should be clear, use the bounding box details effectively, and avoid guessing. Additionally, ensure that the instructions remain strictly relevant to what is visually observable in the satellite image.
    """
    
#     few_shot_examples = [
#     {
#         "role": "user",
#         "content": """Image Caption: 'The image shows a sports complex with two football grounds and one basketball ground.' Coordinates: '40.748817° N, -73.985428° W' Location: 'New York, USA' Labels: 'football: (40.748830°, -73.985440°, 40.748900°, -73.985360°); basketball: (40.748750°, -73.985500°, 40.748800°, -73.985470°)'"""
#     },
#     {
#         "role": "assistant",
#         "content": """
#         Question: Where is the second football ground located relative to the given coordinates?
#         === Answer: The second football ground is slightly northeast of the main location point (40.748817° N, -73.985428° W), bounded by coordinates (40.748830°, -73.985440°, 40.748900°, -73.985360°). It is adjacent to the main field and near a cluster of trees.

#         Question: Can you describe the spatial relationship between the basketball ground and the nearest football field?
#         === Answer: The basketball ground is directly south of the second football field, as shown by its bounding box coordinates (40.748750°, -73.985500°, 40.748800°, -73.985470°). The two are separated by a small patch of green space.
        
#         Question: Using the labels and bounding boxes, what regions are dominant in this area?
#         === Answer: The two football grounds dominate the central area, with the basketball ground occupying the southern region. These sports fields form the primary structures within the given location.
#         """
#     },
#     {
#         "role": "user",
#         "content": """Image Caption: 'The image depicts a coastal area with a marina and several anchored boats.' Coordinates: '36.778259° N, -119.417931° W' Location: 'California, USA' Labels: 'marina: (36.778300°, -119.418000°, 36.778350°, -119.417900°); residential: (36.778250°, -119.417850°, 36.778300°, -119.417800°)'"""
#     },
#     {
#         "role": "assistant",
#         "content": """
#         Question: Where is the marina located in relation to the given coordinates?
#         === Answer: The marina is slightly north of the provided coordinates (36.778259° N, -119.417931° W), located within the bounding box defined by (36.778300°, -119.418000°, 36.778350°, -119.417900°). It is directly adjacent to the shoreline.

#         Question: How is the residential area positioned relative to the marina?
#         === Answer: The residential area is located southeast of the marina, marked by its bounding box coordinates (36.778250°, -119.417850°, 36.778300°, -119.417800°). The two areas are separated by a narrow road visible in the image.
        
#         Question: Can you describe the spatial arrangement of the given labels?
#         === Answer: The marina occupies the northern region near the shoreline, while the residential area is situated to the southeast. The positioning suggests a coastal layout with commercial and residential zoning.
#         """
#     }
# ]
    few_shot_examples = [
    {
        "role": "user",
        "content": """Image Caption: 'The image shows a sports complex with two football grounds and one basketball ground.' Coordinates: '40.748817° N, -73.985428° W' Location: 'New York, USA' Labels: 'football: ((0.52, 0.48), (0.58, 0.52)); basketball: ((0.46, 0.55), (0.48, 0.57))'"""
    },
    {
        "role": "assistant",
        "content": """
        Question: Where is the second football ground located relative to the center coordinates?
        === Answer: The second football ground is slightly northeast of the main location point, bounded by the region ((0.52, 0.48), (0.58, 0.52)). It is adjacent to the main field and near a cluster of trees.

        Question: Can you describe the spatial relationship between the basketball ground and the nearest football field?
        === Answer: The basketball ground is directly south of the second football field. The two are separated by a small patch of green space.

        Question: Using the labels and bounding boxes, what regions are dominant in this area?
        === Answer: The two football grounds dominate the central area, with the basketball ground occupying the southern region. These sports fields form the primary structures within the given location.
        """
    },
    {
        "role": "user",
        "content": """Image Caption: 'The image depicts a coastal area with a marina and several anchored boats.' Coordinates: '36.778259° N, -119.417931° W' Location: 'California, USA' Labels: 'marina: ((0.51, 0.49), (0.53, 0.51)); residential: ((0.49, 0.47), (0.50, 0.48))'"""
    },
    {
        "role": "assistant",
        "content": """
        Question: Where is the marina located in relation to the center coordinates?
        === Answer: The marina is slightly north of the main location point, located near the shoreline within the region ((0.51, 0.49), (0.53, 0.51)).

        Question: How is the residential area positioned relative to the marina?
        === Answer: The residential area is southeast of the marina, separated by a narrow road visible in the image.

        Question: Can you describe the spatial arrangement of the given labels?
        === Answer: The marina occupies the northern part of the scene near the shoreline, while the residential area is situated to the southeast. The layout suggests a coastal zoning with commercial and residential areas.
        """
    }
]


    new_image_data = {
        "role": "user",
        "content": f"Image Caption: '{image_caption}' Coordinates: '{coordinates}' Location: '{location}' Labels: '{labels}'"
    }
    
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",  # Use actual GPT-4 model or correct one
                messages=[{"role": "system", "content": system_prompt}, *few_shot_examples, new_image_data],
                timeout=60  # Timeout to prevent hanging API calls
            )
            return completion.choices[0].message.content
        except RateLimitError as e:
            wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8, 16, ...
            print(f"Rate limit reached. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except OpenAIError as e:
            print(f"API call failed: {e}")
            raise e  # Raise the exception after logging it for debugging purposes

    raise Exception(f"API call failed after {retries} retries.")


def process_row(index, matching_row):
    try:
        # Extract satellite data
        sat_id = str(int(float(matching_row.get('sat_id', '0')))).zfill(8)
        latitude = matching_row.get('latitude', None)
        longitude = matching_row.get('longitude', None)
        location = matching_row.get('state', 'Unknown')
        labels = matching_row.get('pixel_bounds', 'No labels')

        if latitude is None or longitude is None:
            print(f"Skipping row {index}: Missing latitude or longitude.")
            return None

        coordinates = f"{latitude}° N, {longitude}° W"

        # Search for matching satellite data in the summary CSV
        satellite_id_match = sum_csv[sum_csv['SatelliteID'].str.contains(sat_id, regex=False)]
        image_caption = satellite_id_match.iloc[0]['Summarization'] if not satellite_id_match.empty else "No matching summarization found."

        # Generate instructions for the current row
        instructions = generate_instructions(image_caption, coordinates, location, labels)

        # Create result dictionary
        result = {
            "index": index,
            "sat_id": sat_id,
            "coordinates": coordinates,
            "location": location,
            "labels": labels,
            "image_caption": image_caption,
            "instructions": instructions
        }

        # Write the result to the CSV file immediately
        with csv_lock:
            file_exists = os.path.isfile('ins_gen_referring_pixel.csv')
            with open('ins_gen_referring_pixel.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=header)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(result)
                file.flush()
            print(f"Processed row {index} and saved to CSV.")

        return result
    except Exception as e:
        error_log.append(f"Error processing row {index}: {e}")
        print(f"Error processing row {index}: {e}")
        return None

# Load CSV files
df_list = []

# List of CSV files to load
csv_files = [
    '../graft/summarizations/llama-3-8b-instruct/summarizations-llama-3-8b-instruct_batched_naip_0_250_grid.csv',
    '../graft/summarizations/llama-3-8b-instruct/summarizations-llama-3-8b-instruct_batched_naip_6_cont_grid.csv',
    '../graft/summarizations/llama-3-8b-instruct/summarizations-llama-3-8b-instruct_batched_naip_251_475_grid.csv'
]

for file in csv_files:
    df_chunk = pd.read_csv(file)
    df_list.append(df_chunk)

sum_csv = pd.concat(df_list, ignore_index=True)

# Load the main CSV with the region labels
df = pd.read_csv('../Satellite_Eval/all_states_region_labels_with_bounds_pixel_bounds.csv')
print(f"Loaded {len(df)} rows from the CSV file.")

# Filter remaining rows that are not yet processed
existing_coordinates = set()

if os.path.exists('ins_gen_referring_pixel.csv'):
    processed_df = pd.read_csv('ins_gen_referring_pixel.csv', encoding='utf-8')
    existing_coordinates = set(processed_df['coordinates'])

print(f"Found {len(existing_coordinates)} coordinates already processed.")

# Create a coordinates column in the dataframe
df['coordinates'] = df.apply(lambda row: f"{row['latitude']}° N, {row['longitude']}° W", axis=1)

# Filter out rows that have already been processed
remaining_rows = df[~df['coordinates'].isin(existing_coordinates)]

# CSV header
header = ["index", "sat_id", "coordinates", "location", "labels", "image_caption", "instructions"]

# Error log
error_log = []

# Use ThreadPoolExecutor to parallelize the API calls
with ThreadPoolExecutor(max_workers=16) as executor:  # Adjust max_workers as needed
    futures = [executor.submit(process_row, index, row) for index, row in remaining_rows.iterrows()]

    # Use tqdm to show progress
    for future in tqdm(as_completed(futures), total=len(remaining_rows), desc="Processing rows"):
        try:
            future.result()
        except Exception as e:
            print(f"Error processing row: {e}")

# Save errors to a log file
if error_log:
    with open('error_log.txt', 'w') as f:
        for error in error_log:
            f.write(f"{error}\n")

print(f"Results are being saved to ins_gen_referring_pixel.csv in real-time.")