import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Paths to the summarization file, metadata directory, and output file
summarization_file = 'captions/caption_sat_corres_6_cont.csv'
metadata_dir = '/share/kavita/ukm4/datasets/CLIPRS10m_m2o/metadata_db_existing'
output_file = 'ground_image_coordinates.csv'

# Read the summarization CSV file
summarization_df = pd.read_csv(summarization_file)

# Extract the ground image IDs
ground_image_ids = summarization_df['Ground Image Path'].tolist()
ground_image_ids = [int(os.path.splitext(os.path.basename(ground_image_id))[0]) for ground_image_id in ground_image_ids]

# Initialize a set for already processed ground image IDs
processed_image_ids = set()

# Check if the output file exists and read the already processed IDs
if os.path.exists(output_file):
    processed_df = pd.read_csv(output_file)
    processed_image_ids = set(processed_df['ground_image_id'].tolist())
    print(f"Loaded {len(processed_image_ids)} already processed ground image IDs.")

# Filter out already processed ground image IDs
ground_image_ids = [gid for gid in ground_image_ids if gid not in processed_image_ids]
print(f"Processing {len(ground_image_ids)} new ground image IDs.")

# Define a function to check and extract coordinates for a single ground image ID
def check_ground_image_id(ground_image_id, metadata_df):
    # Check if 'latitude' and 'longitude' columns exist
    if ground_image_id in metadata_df.iloc[:, 0].values:
        # Extract the row with the matching ground image ID
        row = metadata_df[metadata_df.isin([ground_image_id]).any(axis=1)]
        # Assuming coordinates are in columns 'latitude' and 'longitude'
        latitude = row.values[4]
        longitude = row.values[3]
        return (ground_image_id, latitude, longitude)
    return None
    

# Define a function to process each metadata file with parallelism for ground image IDs
def process_metadata_file(filename):
    file_path = os.path.join(metadata_dir, filename)
    metadata_df = pd.read_csv(file_path)
    found_coordinates = []

    # Use a ThreadPoolExecutor to parallelize the processing of ground image IDs in the file
    with ThreadPoolExecutor() as executor:
        future_to_gid = {executor.submit(check_ground_image_id, gid, metadata_df): gid for gid in ground_image_ids}
        
        for future in as_completed(future_to_gid):
            result = future.result()
            if result:
                found_coordinates.append(result)
    
    return found_coordinates

# Function to save processed data to the CSV file
def save_processed_data(rows):
    with open(output_file, 'a') as f:
        for row in rows:
            f.write(f"{row[0]},{row[1]},{row[2]}\n")
        f.flush()  # Ensure data is written to disk immediately

# Main logic wrapped in try-except to handle keyboard interruption
try:
    # Create the output CSV file if it doesn't exist and write the header
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            f.write('ground_image_id,latitude,longitude\n')

    # Create a thread pool and process the metadata files in parallel
    with ThreadPoolExecutor() as executor:
        futures = []
        total_rows_processed = 0  # Counter to track the total number of rows processed

        # Submit each metadata file for processing
        for filename in os.listdir(metadata_dir):
            if filename.endswith('.csv'):
                futures.append(executor.submit(process_metadata_file, filename))

        # Collect the results as they are completed and save them in real-time
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing metadata files"):
            results = future.result()  # Get the rows from the completed future
            if results:
                total_rows_processed += len(results)  # Count how many rows this future processed
                save_processed_data(results)  # Save the processed rows
                print(f"Rows processed so far: {total_rows_processed}")  # Display progress

except KeyboardInterrupt:
    print("Keyboard interrupt received. Saving in-progress futures...")

    # Collect the results of completed futures up to the interruption point
    for future in futures:
        if future.done():  # Check if the future has finished before the interrupt
            results = future.result()
            if results:
                save_processed_data(results)  # Save results from completed futures
                print(f"Rows saved from interrupted processing.")

finally:
    print(f"Total rows processed: {total_rows_processed}. Coordinates saved successfully.")
