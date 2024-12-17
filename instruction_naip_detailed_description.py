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

output_file = 'ins_gen_detailed_description.csv'

def generate_instructions(image_caption, coordinates, location, labels, retries=5):
    # system_prompt = """
    # You are an AI assistant specialized in analyzing satellite images. You will be given a caption of a satellite image, along with the coordinates, location name, and a list of labels with their corresponding bounding boxes (the list may be empty, indicating that the satellite image has no labels). Based on this information, generate a series of questions and answers about the image. The questions should cover the visual content and relevant geographical information. Ensure that all questions provide detailed and well-reasoned answers. Avoid inferring or mentioning anything not directly visible.
    # """
    system_prompt = """
    You are an AI visual assistant that can analyze a single image. You receive several sentences, each describing the same satellite image you are observing. In addition, specific object locations within the image are given, along with detailed coordinates. One label of the object might have multiple bounding boxes, resulting in a list of bounding boxes contained by '[]'. These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.

    Using the provided caption and bounding box information, describe the scene in a detailed manner. Please ignore the images that cannot be observed from the satellite view. The satellite image is taken from a top-down perspective, and the region is approximately 100 m x 100 m.

    Instead of directly mentioning the bounding box coordinates, utilize this data to explain the scene using natural language. Include details like object counts, position of the objects, relative position between the objects.

    When using the information from the caption and coordinates, directly explain the scene, and do not mention that the information source is the caption or the bounding box.  Always answer as if you are directly looking at the image.
    """
    
    # Few-shot examples remain the same
    # few_shot_examples = [
    #     {
    #         "role": "user",
    #         "content": """Image Caption: 'The image show residential houses with small gardens. The roads are narrow and lined with parked cars. Trees and greenery are scattered throughout the area, providing shade and a suburban feel. The area seems quiet, with little visible commercial activity.' Coordinates: '40.7128° N, 74.0060° W' Location: 'Kearny, New Jersey, USA' Labels: 'residential: (40.7130°, -74.0070°, 40.7150°, -74.0050°)'"""
    #     },
    #     {
    #         "role": "assistant",
    #         "content": """Question: What does the satellite image reveal about the area based on the ground-level information? 
    #         === Answer: The satellite image likely shows a residential neighborhood with low-rise buildings, small gardens, and a network of narrow roads, in line with the ground-level view of houses and suburban greenery.
            
    #         Question: How does the presence of trees and greenery at ground level translate to the satellite view? 
    #         === Answer: The trees and greenery visible from the ground would likely appear as small green patches or clusters in the satellite image, offering a balance between urban development and natural areas.
            
    #         Question: Is there any evidence of commercial zones nearby? 
    #         === Answer: Based on the caption and visible residential nature, there is no indication of significant commercial zones in the immediate vicinity of the satellite image.
            
    #         Question: What can the coordinates and location tell us about the broader geographical context? 
    #         === Answer: These coordinates place the location in a suburban area of New Jersey, likely a quieter residential zone.
            
    #         Question: What labels are associated with this image? 
    #         === Answer: The labels provided include 'residential', marking areas of residential buildings in the satellite image. 
    #         """
    #     },
    #     {
    #         "role": "user",
    #         "content": """Image Caption: 'The ground images show a densely populated urban area with a grid-like street system and numerous mid-rise and high-rise buildings. The area features a cluster of skyscrapers, which suggests a major downtown district. There is significant vehicular traffic on major streets. A large green park area is visible nearby.' Coordinates: '41.8821° N, 87.6256° W' Location: 'Chicago, Illinois, USA' Labels: 'commercial: (41.8790°, -87.6305°, 41.8770°, -87.6285°); park: (41.8800°, -87.6290°, 41.8820°, -87.6270°)'"""
    #     },
    #     {
    #         "role": "assistant",
    #         "content": """Question: What does the satellite image indicate about the urban structure of the city? 
    #         === Answer: The satellite image would likely reveal a densely populated urban area with a grid-like street pattern, and a visible cluster of skyscrapers in the central area, which suggests the downtown or commercial district.
            
    #         Question: How does the park described in the ground images relate to the satellite view? 
    #         === Answer: The park would likely appear as a large green area near the center of the image, offering a contrast to the dense urban development around it.
            
    #         Question: What can be inferred about human activity and infrastructure from the ground images? 
    #         === Answer: The ground images suggest significant vehicular traffic and visible infrastructure like mid-rise and high-rise buildings. The satellite image might capture these features indirectly through denser urban layouts and areas of higher development.
            
    #         Question: How do the skyscrapers described at ground level affect the satellite image? 
    #         === Answer: The skyscrapers would be prominent in the satellite image, casting large shadows over the surrounding areas and occupying more space compared to lower buildings.
            
    #         Question: What labels are associated with this image? 
    #         === Answer: The labels provided include 'commercial' for the downtown area and 'park' for the green space in the image."""
    #     }
    # ]
    
    few_shot_examples = [
    {
        "role": "user",
        "content": """Image Caption: 
        - The image shows residential houses with small gardens. 
        - The roads are narrow and lined with parked cars. 
        - Trees and greenery are scattered throughout the area, providing shade and a suburban feel. 
        - The area seems quiet, with little visible commercial activity. 
        Coordinates: '40.7128° N, 74.0060° W' 
        Location: 'Kearny, New Jersey, USA' 
        Labels: 'residential: (0.45, 0.32, 0.55, 0.37)'"""
    },
    {
        "role": "assistant",
        "content": """Description:
        The satellite image depicts a quiet residential neighborhood with small houses surrounded by individual gardens and greenery. The narrow roads lined with parked cars indicate a suburban environment, primarily designed for residential purposes rather than commercial. Trees and vegetation create shaded areas within the neighborhood, enhancing the peaceful, suburban character. The coordinates point to a location in Kearny, New Jersey, USA, suggesting a suburban town setting near New York City. Labels mark 'residential' areas, emphasizing the non-commercial nature of the neighborhood and identifying the core residential clusters.
        """
    },
    {
        "role": "user",
        "content": """Image Caption: 
        - The ground images show a densely populated urban area with a grid-like street system and numerous mid-rise and high-rise buildings. 
        - The area features a cluster of skyscrapers, which suggests a major downtown district. 
        - There is significant vehicular traffic on major streets. 
        - A large green park area is visible nearby. 
        Coordinates: '41.8821° N, 87.6256° W' 
        Location: 'Chicago, Illinois, USA' 
        Labels: 'commercial: (0.30, 0.20, 0.35, 0.25); park: (0.32, 0.22, 0.40, 0.27)'"""
    },
    {
        "role": "assistant",
        "content": """Description:
        This satellite image illustrates a densely packed urban area with a well-organized, grid-like street layout, typical of Chicago's downtown. A cluster of skyscrapers dominates the central area, identifying it as a major commercial hub with high-rise buildings, indicating the financial district or downtown core. Significant vehicular movement along main roads reveals a busy, active city center. Nearby, a large green park offers open space amid the dense urban surroundings, standing out as a clear, rectangular green area. The provided labels denote 'commercial' zones for the skyscraper cluster and 'park' for the green space, emphasizing the mix of business and recreational areas within the urban layout.
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
            file_exists = os.path.isfile(output_file)
            with open(output_file, mode='a', newline='', encoding='utf-8') as file:
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

if os.path.exists(output_file):
    processed_df = pd.read_csv(output_file, encoding='utf-8')
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
with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers as needed
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

print(f"Results are being saved to {output_file} in real-time.")