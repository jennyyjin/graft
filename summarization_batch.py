import pandas as pd
import torch
import transformers
from transformers import LlamaForCausalLM, AutoTokenizer
from tqdm import tqdm
import csv
import os
import pickle
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
# file_path = 'captions/caption_sat_corres_251_475.csv'
# data = pd.read_csv(file_path)

# # print(data.columns)

# # Define a function to merge rows with the same SatelliteID and concatenate the captions
# def merge_and_concatenate(df):
#     # Group by 'Satellite Image Path'
#     grouped = df.groupby('Satellite Image Path')

#     # Function to concatenate captions and ground image paths
#     def concatenate_data(group):
#         # Remove line breaks from captions and concatenate them
#         captions = [caption.replace('\n', ' ') for caption in group['Caption']]
#         captions = "\n".join([f"{i+1}. {caption}" for i, caption in enumerate(captions)])

#         # Concatenate ground image paths as comma-separated strings
#         ground_image_ids = ", ".join(group['Ground Image Path'].astype(str))

#         # Return the concatenated result as a series
#         return pd.Series({
#             'SatelliteID': group.name,  # use group name, which is 'Satellite Image Path'
#             'Captions': captions,
#             'GroundImageIDs': ground_image_ids
#         })

#     # Apply the function to each group and reset the index
#     merged_data = grouped.apply(concatenate_data).reset_index(drop=True)
    
#     # print(merged_data.columns)

#     return merged_data

# # Apply the function to the data
# df = merge_and_concatenate(data)
# # df.to_csv('captions/merged_caption_sat_corres_6_cont.csv')
csv_name = '251_475'
csv_order = '3'
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
    north_boundary = center_lat + halfwidth
    south_boundary = center_lat - halfwidth
    east_boundary = center_lon + halfwidth
    west_boundary = center_lon - halfwidth
    
    if lat > center_lat:
        if lon > center_lon:
            return 'northeast'
        elif lon < center_lon:
            return 'northwest'
        else:
            return 'north'
    elif lat < center_lat:
        if lon > center_lon:
            return 'southeast'
        elif lon < center_lon:
            return 'southwest'
        else:
            return 'south'
    else:
        if lon > center_lon:
            return 'east'
        elif lon < center_lon:
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
            halfwidth = 0.05  # Example halfwidth value, adjust according to your dataset
            
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

# exit()

# Add a new column for summarization if it doesn't exist
if 'Summarization' not in df.columns:
    df['Summarization'] = ""

# print(df.columns)

# Output CSV file path
output_dir = 'summarizations/llama-3-8b-instruct'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'summarizations-llama-3-8b-instruct_batched_naip_251_475_debug2.csv')

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
    
prompt_template = """
    You are a helpful AI assistant for summarizing image descriptions from the same region taken by a satellite.

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
        - **Signs/Billboards**: Ignore unless they suggest a specific type of area (e.g., commercial or industrial), if they suggest a specific type of area, also do not mention the object itself, just the property of the area.

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
    1. This image is taken in the north of the satellite region. The image depicts a lush green field with a tree in the foreground. The tree appears barren, with no leaves on its branches. In the field, there are two cows grazing on the grass.

    2. This image is taken in the northwest of the satellite region. The image features a lush green field with trees in the background. The sky above is blue and cloudy. There are no people or animals visible, emphasizing the natural beauty of the landscape.
    ]

    **Summary (Bullet Points):**

    - The region, primarily in the northern part, is characterized by a lush green field.
    - Trees are present in the background, particularly in the northwest, indicating possible farmland.
    - Large open spaces suggest agricultural or rural land use.

    ---

    **Correct Example 2:**

    Captions:

    [
    1. This image is taken in the east of the satellite region. The image shows a crowded street with people walking and cars parked on the side. There are shops and small restaurants visible along the street.

    2. This image is taken in the southeast of the satellite region. The image features a food truck parked beside a building, with people standing in line. Some small plants are visible near the truck.
    ]
    **Summary (Bullet Points):**

    - The region, primarily in the eastern and southeastern parts, is an urban area.
    - Small commercial establishments like shops and restaurants are visible in both sections.
    - The presence of transportation infrastructure such as streets and parked cars is evident throughout.

    ---

    **Incorrect Example 1:**

    Captions:

    [
    1. This image is taken in the center of the satellite region. The image features a large brick tower with a red and black color scheme. The tower stands prominently in the foreground, towering over the surrounding landscape. The sky serves as a backdrop for the impressive structure, enhancing its presence in the scene.

    2. This image is taken in the south of the satellite region. The image features a large red and black lighthouse towering over the surrounding area. The lighthouse is prominently visible in the scene, with its vibrant colors standing out against the backdrop. The sky above the lighthouse appears to be clear and blue, adding to the serene atmosphere of the scene.
    ]
    **Incorrect Summary (Bullet Points):**

    - The region is characterized by large brick towers, possibly lighthouses, located in the center and southern parts.
    - Some areas feature grassy spaces, houses, cars, and clock towers, suggesting a mix of historical and residential areas.
    - Stained glass windows and other small architectural details are visible.
    - The overall region includes a combination of urban and natural environments.

    **Reason for Incorrectness:**

    - Should not mention atmosphere or ambience (such as the sky or serene atmosphere).
    - Describing multiple scenes instead of summarizing the region as a whole.
    - References to small architectural details like stained glass windows are not visible from a satellite view.
    
    Corrected Summary (Bullet Points):

    - The region is characterized by large brick towers, possibly lighthouses, visible in both the central and southern parts of the satellite region.
    - The towers are prominent landmarks and stand out in the surrounding landscape.
    - The surrounding area includes open land with few significant structures.
    ---

    **Incorrect Example 2:**

    Captions:

    [
    1. This image is taken in the northwest of the satellite region. The image features a train traveling down the tracks. The train is quite large and occupies a significant portion of the scene. The train appears to be passing through a countryside area, with trees visible in the background. Additionally, there is a crane in the foreground of the image, adding an interesting element to the scene.
    ]

    **Incorrect Summary (Bullet Points):**

    - The region features train tracks running through a countryside area in the northwest.
    - The presence of a crane in the foreground suggests industrial or construction activity.
    - Trees in the background suggest a rural environment.

    **Reason for Incorrectness:**

    - Should not mention transient objects like the crane directly.
    - The summary should focus on larger-scale features like transportation infrastructure (the train track) and the countryside.

    Corrected Summary (Bullet Points):

    - The region in the northwest features train tracks running through a countryside area.
    - Possible industrial or construction activities inferred from nearby infrastructure.
    - Trees in the background suggest a rural environment.

    ---

    Your Task:

    Now, please summarize the following descriptions within the brackets:

    [{captions}]

    **Summary (Bullet Points):**
    """


batch_size = 1  # Define batch size
prompts = []
rows_to_process = []

# Create prompts and batches
for index, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Processing rows"):
    captions = row['Captions']
    # messages = [
    #     {"role": "system", "content": "You are a helpful AI assistant for summarizing image descriptions from the same region taken by a satellite."},
    #     {"role": "user", "content": f"The texts in the brackets are descriptions \
    #         of images taken in the same region, ordered numerically without preference. \
    #         Summarize these descriptions to provide an overview of the region. \
    #         Avoid unnecessary preambles and transient objects, and also think from an ariel level, \
    #         so ignore the ground level detail that cannot be seen from a satellite's point of view. \
    #         For example, if you see people and wildlives in the captions, do not include those explicitly, but infer to what this might possibly mean, \
    #         like human indicates human activities, which means the region would be a site of attraction, and cows would indicate that the region might be a farming field.\
    #         Do not suggest about any ambience about the region, but focus on the physical attributes. \
    #         For example, if you see a caption mentioning a lush green field with tranquil atmosphere, you can suggest that the region \
    #         is characterized by a lush green field. \
    #         Here are some examples: \n\n     \
    #         Example1:\n[1. The image depicts a lush green field with a tree in \
    #         the foreground. The tree appears to be barren, with no leaves on \
    #         its branches. In the field, there are two cows, one closer to \
    #         the foreground and the other further back. The cows are grazing \
    #         on the grass, creating a serene and peaceful atmosphere in the scene. \
    #         2. The image features a lush green field with trees in the background. \
    #         The sky above the field is blue and cloudy, creating a serene atmosphere. \
    #         There are no people or animals visible in the scene, emphasizing the \
    #         natural beauty of the landscape.] \n \
    #         <Summary>: The region is characterized by a lush green field, surrounded \
    #         by trees. It might involve farms or golf court based on the area of the field.  \n\n\
    #         Example2:\n[1. The image features a man in a yellow safety jacket working \
    #         on a crane. The crane is positioned above a business sign, possibly \
    #         a restaurant. The man appears to be fixing the crane, ensuring its \
    #         proper functioning.  In the scene, there is a truck parked near the \
    #         crane, and a bench can be seen on the right side of the image. Additionally, \
    #         there is a traffic light visible in the background, adding to the urban \
    #         setting of the scene.\n2. The image features a small restaurant situated \
    #         in a parking lot. The restaurant has a red roof and is surrounded by several \
    #         cars parked nearby. There is also a truck parked in the vicinity of the restaurant.  \
    #         In addition to the cars and the truck, there is a bench and a dining table visible \
    #         in the scene. The bench is placed close to the restaurant, while the dining table \
    #         is located further away, closer to the edge of the parking lot.\n3. The image \
    #         features a food truck parked on the side of a street. There are two people standing \
    #         in front of the food truck, likely waiting to be served. One person is positioned \
    #         closer to the left side of the truck, while the other person is standing a bit\
    #         further to the right.  The food truck appears to be selling Mexican food, \
    #         as evidenced by the presence of a bowl and a spoon in the scene. The bowl \
    #         is placed near the center of the truck, while the spoon is located closer \
    #         to the right side of the truck.\n4. The image features a train traveling \
    #         down the train tracks. The train is quite long, occupying a significant \
    #         portion of the scene. The tracks are located next to a body of water, \
    #         creating a picturesque backdrop for the train's journey. The sky above \
    #         the scene is cloudy, adding to the overall atmosphere of the image.\n5. \
    #         The image features a delicious meal consisting of a hamburger and a side \
    #         of french fries. The hamburger is placed in the center of the scene, while \
    #         the french fries are scattered around it. The hamburger appears to be a \
    #         cheeseburger, adding to the appetizing nature of the meal.] \n \
    #         <Summary>: The region in the image aligns with an urban area featuring \
    #         small commercial and food establishments. The presence of buildings with \
    #         flat roofs in a grid layout suggests businesses, potentially including a \
    #         small restaurant with a parking lot as described. There are visible train tracks running \
    #         alongside a body of water. The parking spaces around some buildings \
    #         could correspond to locations where food trucks or outdoor dining \
    #         spaces might be set up, indicating a mix \
    #         of commercial and open areas for social activity. Overall, the region \
    #         combines urban commercial structures with transportation infrastructure. \n\n\
    #         Here's a wrong example:\nCaptions: [1. The image features a large brick tower with a red and black color scheme. The tower stands prominently in the foreground, towering over the surrounding landscape. The sky serves as a backdrop for the impressive structure, enhancing its presence in the scene.\n2. The image features a large red and black lighthouse towering over the surrounding area. The lighthouse is prominently visible in the scene, with its vibrant colors standing out against the backdrop. The sky above the lighthouse appears to be clear and blue, adding to the serene atmosphere of the scene.\n3. The image features a large brick tower, possibly a lighthouse, towering over the surrounding landscape. The tower is prominently positioned in the foreground of the scene. In the background, there is a house and a car, giving a sense of scale to the impressive brick structure. Additionally, there is a bird flying in the sky, adding a touch of life to the otherwise static scene.\n4. The image features a large brick building with a clock tower on top. The clock tower is prominently visible, towering over the rest of the building. The building appears to be made of red bricks, giving it an old and historical appearance. The clock tower stands out as the main focal point of the scene.\n5. The image features a large brick tower with a red top, towering over the surrounding area. The tower has a clock on its side, making it a prominent landmark. The blue sky serves as a beautiful backdrop for the tower, enhancing its presence in the scene.\n6. The image features a large brick tower, possibly a lighthouse, towering over a grassy area. The tower has a red and black color scheme, making it stand out against the backdrop. The sky above the tower appears to be cloudy, adding to the overall atmosphere of the scene.\n7. The image features a large brick building with a prominent arched doorway. The door is open, revealing the interior of the building. The doorway is adorned with a stained glass window, adding an artistic touch to the scene. The building appears to be made of red bricks, giving it a warm and rustic appearance.\n8. The image features two large trees standing next to each other in a grassy area. The trees are surrounded by a mix of grass and dirt, creating a natural and serene environment. The trees are the main focus of the scene, and their presence adds a sense of tranquility to the landscape.\n9. The image features a large brick tower, possibly a lighthouse, towering over the surrounding area. The tower appears to be made of red bricks, giving it a distinctive appearance. The sky serves as a beautiful backdrop for the tower, highlighting its prominence in the scene.\n10. The image features a large brick tower with a red top, towering over the surrounding area. The tower appears to be made of bricks, giving it an old-fashioned look. The sky in the background is partly cloudy, adding to the overall atmosphere of the scene. \
    #         <Summary>: The region in the images is characterized by the presence of large brick towers, possibly lighthouses, with a red and black color scheme. The towers stand prominently in the foreground, often towering over the surrounding landscape. The scenes also feature a mix of natural and urban elements, including grassy areas, houses, cars, and buildings with flat roofs. The presence of clock towers and stained glass windows suggests a blend of historical and architectural features. The overall atmosphere is serene, with clear blue skies in some scenes and cloudy skies in others. The region appears to be a combination of natural and urban areas, possibly with a focus on historical or architectural landmarks.  \
    #         Reason that this is a wrong example for summary: The sentence ' The overall atmosphere is serene, with clear blue skies in some scenes and cloudy skies in others.' is wrong. Firstly, it should not mention the atmosphere, and secondly, in a region, it would not have multiple scenes. In the description, imaging you are generalizing the description for the same region from multiple images, but eventually you are describing just one region. The sentence 'The presence of clock towers and stained glass windows suggests a blend of historical and architectural features.' is also wrong because it observes the glasses, which is invisible form a satellite ariel view. \
    #         The more correct summarization should be:\n <Summary>: The region in the images is characterized by the presence of large brick towers, possibly lighthouses, with a red and black color scheme. The towers stand prominently in the foreground, often towering over the surrounding landscape. The scenes also feature a mix of natural and urban elements, including grassy areas, houses, cars, and buildings with flat roofs. The presence of clock towers suggests a blend of historical and architectural features. The region appears to be a combination of natural and urban areas, possibly with a focus on historical or architectural landmarks.  \
    #         Now, please summarize the following descriptions in the bracket: \n\n     \
    #         [{captions}] \n <Summary>:"}
    # ]
    
    # long response version
    # messages = [
    #     {
    #         "role": "system",
    #         "content": "You are a helpful AI assistant for summarizing image descriptions from the same region taken by a satellite."
    #     },
    #     {
    #         "role": "user",
    #         "content": f"\nThe texts within the brackets are descriptions of images taken in the same region, numbered sequentially without preference. Your task is to combine these descriptions to provide an accurate, high-level overview of the region based on satellite imagery. **Do not include or explicitly mention any transient objects or small-scale details that would not be visible from a satellite\u2019s perspective.**\n\nGuidelines:\n\n1. **Aerial Perspective:**\n   - Summarize from a satellite's point of view.\n   - **Ignore any ground-level details** such as:\n     - Individual people, animals (like dogs, cows, birds, or wildlife), small vehicles (bicycles), or other small objects.\n     - Small architectural details (e.g., windows, decorations) and any specific action happening on the ground level.\n\n2. **Infer Rather Than Describe:**\n   - If people, animals, or transient objects appear in the captions, **do not include them directly**. Instead, infer what their presence might indicate about the region:\n     - **People**: Suggest human activities or an urban/commercial area.\n     - **Cows**: Indicate farmland or agricultural land.\n     - **Birds/Wildlife**: Ignore unless they provide clues about the environment (e.g., wildlife could hint at a natural habitat).\n\n3. **Focus on Permanent, Physical Attributes:**\n   - Describe only large-scale and permanent features that are visible from above (e.g., fields, buildings, bodies of water, transportation infrastructure like roads and railways).\n   - Avoid any mention of atmospheric conditions or transient qualities like a \"peaceful\" or \"busy\" atmosphere.\n   - For example, if a caption mentions a \"crowded area with many people,\" summarize as \"an urban or commercial area.\"\n\n4. **Emphasize Consistency Across Images:**\n   - Summarize as if you are describing a single cohesive region. Focus on recurring features and ignore any small, non-permanent elements that vary from image to image.\n\nExamples:\n\n---\n\n**Correct Example 1:**\n\nCaptions:\n\n[\n1. The image depicts a lush green field with a tree in the foreground. The tree appears barren, with no leaves on its branches. In the field, there are two cows grazing on the grass.\n\n2. The image features a lush green field with trees in the background. The sky above is blue and cloudy. There are no people or animals visible, emphasizing the natural beauty of the landscape.\n]\n\nSummary:\n\nThe region is characterized by a lush green field surrounded by trees. It might involve farms or a golf course based on the expanse of the field.\n\n---\n\n**Correct Example 2:**\n\nCaptions:\n\n[\n1. The image shows a crowded street with people walking and cars parked on the side. There are shops and small restaurants visible along the street.\n\n2. The image features a food truck parked beside a building, with people standing in line. Some small plants are visible near the truck.\n]\n\nSummary:\n\nThe region aligns with an urban area featuring small commercial and food establishments. Buildings with flat roofs suggest businesses like restaurants, and the presence of vehicles indicates transportation infrastructure.\n\n---\n\n**Incorrect Example 1:**\n\nCaptions:\n\n[\n1. The image features a large brick tower with a red and black color scheme. The tower stands prominently in the foreground, towering over the surrounding landscape. The sky serves as a backdrop for the impressive structure, enhancing its presence in the scene.\n\n2. The image features a large red and black lighthouse towering over the surrounding area. The lighthouse is prominently visible in the scene, with its vibrant colors standing out against the backdrop. The sky above the lighthouse appears to be clear and blue, adding to the serene atmosphere of the scene.\n\n3. The image features a large brick tower, possibly a lighthouse, towering over the surrounding landscape. The tower is prominently positioned in the foreground of the scene. In the background, there is a house and a car, giving a sense of scale to the impressive brick structure. Additionally, there is a bird flying in the sky, adding a touch of life to the otherwise static scene.\n\n4. The image features a large brick building with a clock tower on top. The clock tower is prominently visible, towering over the rest of the building. The building appears to be made of red bricks, giving it an old and historical appearance. The clock tower stands out as the main focal point of the scene.\n\n5. The image features a large brick tower with a red top, towering over the surrounding area. The tower has a clock on its side, making it a prominent landmark. The blue sky serves as a beautiful backdrop for the tower, enhancing its presence in the scene.\n\n6. The image features a large brick tower, possibly a lighthouse, towering over a grassy area. The tower has a red and black color scheme, making it stand out against the backdrop. The sky above the tower appears to be cloudy, adding to the overall atmosphere of the scene.\n\n7. The image features a large brick building with a prominent arched doorway. The door is open, revealing the interior of the building. The doorway is adorned with a stained glass window, adding an artistic touch to the scene. The building appears to be made of red bricks, giving it a warm and rustic appearance.\n\n8. The image features two large trees standing next to each other in a grassy area. The trees are surrounded by a mix of grass and dirt, creating a natural and serene environment. The trees are the main focus of the scene, and their presence adds a sense of tranquility to the landscape.\n\n9. The image features a large brick tower, possibly a lighthouse, towering over the surrounding area. The tower appears to be made of red bricks, giving it a distinctive appearance. The sky serves as a beautiful backdrop for the tower, highlighting its prominence in the scene.\n\n10. The image features a large brick tower with a red top, towering over the surrounding area. The tower appears to be made of bricks, giving it an old-fashioned look. The sky in the background is partly cloudy, adding to the overall atmosphere of the scene.\n]\n\nIncorrect Summary:\n\nThe region is characterized by large brick towers, possibly lighthouses, with a red and black color scheme. Scenes feature natural and urban elements like grassy areas, houses, cars, and buildings with flat roofs. The presence of clock towers and stained glass windows suggests historical and architectural features. The overall atmosphere is serene, with clear and cloudy skies. The region appears to be a combination of natural and urban areas focusing on historical landmarks.\n\n**Reason for Incorrectness:**\n- Atmosphere Mentioned: Should not mention ambience or atmospheric conditions.\n- Multiple Scenes: Describing multiple scenes instead of summarizing the region as a whole.\n- Invisible Details: References to stained glass windows are not visible from a satellite view.\n\n---\n\n**Incorrect Example 2:**\n\nCaptions:\n\n[\n1. The image features a train traveling down the tracks. The train is quite large and occupies a significant portion of the scene. The train appears to be passing through a countryside area, with trees visible in the background. Additionally, there is a crane in the foreground of the image, adding an interesting element to the scene.\n]\n\nIncorrect Summary:\n\nThe region is characterized by a train track running through a countryside area with trees in the background. The presence of a crane in the foreground suggests possible construction or industrial activities.\n\n**Reason for Incorrectness:**\n- Foreground Mentioned: The crane is a ground-level detail and should not be included when summarizing the region. The summary should focus on construction or industrial activities without specifying the crane.\n\nCorrected Summary:\n\nThe region is characterized by a train track running through a countryside area with trees. The area suggests possible construction or industrial activities.\n\n---\n\nYour Task:\n\nNow, please summarize the following descriptions within the brackets:\n\n[{captions}]\n\n<Summary>:\n"
    #     }
    # ]
    
    # bulletpoint version
    # messages = [
    #     {
    #     "role": "system",
    #     "content": "You are a helpful AI assistant for summarizing image descriptions from the same region taken by a satellite."
    #     },
    #     {
    #     "role": "user",
    #     "content": f"\nThe texts within the brackets are descriptions of images taken in the same region, numbered sequentially without preference. Your task is to combine these descriptions to provide an accurate, high-level overview of the region based on satellite imagery. **Do not include or explicitly mention any transient objects or small-scale details that would not be visible from a satellite’s perspective.**\n\nGuidelines:\n\n1. **Aerial Perspective:**\n   - Summarize from a satellite's point of view.\n   - **Ignore any ground-level details** such as:\n     - Individual people, animals (like dogs, cows, birds, or wildlife), small vehicles (bicycles), or other small objects.\n     - Small architectural details (e.g., windows, decorations) and any specific action happening on the ground level.\n\n2. **Infer Rather Than Describe:**\n   - If people, animals, or transient objects appear in the captions, **do not include them directly**. Instead, infer what their presence might indicate about the region:\n     - **People**: Suggest human activities or an urban/commercial area.\n     - **Cows**: Indicate farmland or agricultural land.\n     - **Birds/Wildlife**: Ignore unless they provide clues about the environment (e.g., wildlife could hint at a natural habitat).\n\n3. **Focus on Permanent, Physical Attributes:**\n   - Describe only large-scale and permanent features that are visible from above (e.g., fields, buildings, bodies of water, transportation infrastructure like roads and railways).\n   - Avoid any mention of atmospheric conditions or transient qualities like a \"peaceful\" or \"busy\" atmosphere.\n   - For example, if a caption mentions a \"crowded area with many people,\" summarize as \"an urban or commercial area.\"\n\n4. **Emphasize Consistency Across Images:**\n   - Summarize as if you are describing a single cohesive region. Focus on recurring features and ignore any small, non-permanent elements that vary from image to image.\n\n5. **Use Bullet Points:**\n   - Present the summarized details in bullet points for clarity and conciseness.\n\nExamples:\n\n---\n\n**Correct Example 1:**\n\nCaptions:\n\n[\n1. The image depicts a lush green field with a tree in the foreground. The tree appears barren, with no leaves on its branches. In the field, there are two cows grazing on the grass.\n\n2. The image features a lush green field with trees in the background. The sky above is blue and cloudy. There are no people or animals visible, emphasizing the natural beauty of the landscape.\n]\n\n**Summary (Bullet Points):**\n\n- The region is characterized by a lush green field.\n- Trees are present in the background, indicating possible farmland.\n- Large open spaces suggest agricultural or rural land use.\n\n---\n\n**Correct Example 2:**\n\nCaptions:\n\n[\n1. The image shows a crowded street with people walking and cars parked on the side. There are shops and small restaurants visible along the street.\n\n2. The image features a food truck parked beside a building, with people standing in line. Some small plants are visible near the truck.\n]\n\n**Summary (Bullet Points):**\n\n- The region is an urban area.\n- Small commercial establishments like shops and restaurants are visible.\n- The presence of transportation infrastructure such as streets and parked cars.\n\n---\n\n**Incorrect Example 1:**\n\nCaptions:\n\n[\n1. The image features a large brick tower with a red and black color scheme. The tower stands prominently in the foreground, towering over the surrounding landscape. The sky serves as a backdrop for the impressive structure, enhancing its presence in the scene.\n\n2. The image features a large red and black lighthouse towering over the surrounding area. The lighthouse is prominently visible in the scene, with its vibrant colors standing out against the backdrop. The sky above the lighthouse appears to be clear and blue, adding to the serene atmosphere of the scene.\n]\n\n**Incorrect Summary (Bullet Points):**\n\n- The region is characterized by large brick towers, possibly lighthouses.\n- Some areas feature grassy spaces, houses, cars, and clock towers, suggesting a mix of historical and residential areas.\n- Stained glass windows and other small architectural details are visible.\n- The overall region includes a combination of urban and natural environments.\n\n**Reason for Incorrectness:**\n- Should not mention atmosphere or ambience (such as the sky or serene atmosphere).\n- Describing multiple scenes instead of summarizing the region as a whole.\n- References to small architectural details like stained glass windows are not visible from a satellite view.\n\n---\n\n**Incorrect Example 2:**\n\nCaptions:\n\n[\n1. The image features a train traveling down the tracks. The train is quite large and occupies a significant portion of the scene. The train appears to be passing through a countryside area, with trees visible in the background. Additionally, there is a crane in the foreground of the image, adding an interesting element to the scene.\n]\n\n**Incorrect Summary (Bullet Points):**\n\n- The region features train tracks running through a countryside area.\n- The presence of a crane in the foreground suggests industrial or construction activity.\n- Trees in the background suggest a rural environment.\n\n**Reason for Incorrectness:**\n- Should not mention transient objects like the crane directly.\n- The summary should focus on larger-scale features like transportation infrastructure (the train track) and the countryside.\n\nCorrected Summary (Bullet Points):\n\n- The region features train tracks running through a countryside area.\n- Possible industrial or construction activities inferred from nearby infrastructure.\n- Trees in the background suggest a rural environment.\n\n---\n\nYour Task:\n\nNow, please summarize the following descriptions within the brackets:\n\n[{captions}]\n\n**Summary (Bullet Points):**\n"
    #     }
    # ]
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
                    print(generated_text)
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
