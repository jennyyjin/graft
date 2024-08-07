# create a script that do the following tasks:
# 1. get the rows from the captions.csv file, extract the image path
# 2. the image path is in this form, extract the 25374260089.jpg from ../graft/data/images_processed/0/0/8/9/25374260089.jpg for example, call it satid
# 3. use the satid to get the image ids from the satellite_centers.pkl file
# 4. assocaite the caption for each image to the corresponding satellite image
# 5. pass it to llama using vllm to promt for a summarization of the satellite image
# 6. save the summarization in a file
# 7. repeat for all the images

# #############################################################################
import csv
import pickle
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import os
import torch
from os.path import isfile
# from rouge_score import rouge_scorer

# os.environ['TORCH_USE_CUDA_DSA'] = '1'

# llm = LLM(model="openlm-research/open_llama_13b", enforce_eager=True)
# sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens = 32)

def extract_gid(image_path):
    match = re.search(r'(\d+)\.jpg$', image_path)
    if match:
        return match.group(1)
    return None

def read_captions(file_path):
    """
    get a dictionary with the ground image id as key and the caption as value
    """
    captions = {}
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row if exists
        for row in tqdm(reader, desc="Reading captions"):
            image_path, caption = row
            gid = extract_gid(image_path)
            if gid:
                captions[int(gid)] = caption.replace('\n', ' ').strip()
    return captions

def read_satellite_centers(file_path):
    with open(file_path, 'rb') as pklfile:
        satellite_centers = pickle.load(pklfile)
    return satellite_centers

# def prompt_llama(captions):
#     # formatted_captions = "\n".join([f"Caption {i+1}: {caption}" for i, caption in enumerate(captions)])
#     # prompt = (
#     #     f"Below are several image descriptions taken within a common region. "
#     #     "Please summarize the following descriptions to generate a comprehensive summary of the region.\n\n"
#     #     f"{formatted_captions}\n\n"
#     #     "Summarization:"
#     # )
#     formatted_captions = " ".join(captions)
#     prompt = (
#         f"Summarize the text in bracket."
#         f"[{formatted_captions}]"
#     )
#     print(prompt)
#     with torch.cuda.amp.autocast():
#         result = llm.generate([prompt], sampling_params=sampling_params)
    
#     generated_text = result[0].outputs[0].text.strip() 
#     return generated_text

# def save_summarization(satid, summarization, data, csv_writer):
#     captions = "\n".join([f"{i+1}. {caption}" for i, caption in enumerate(data['captions'])])
#     ground_image_ids = ", ".join(map(str, data['ground_image_ids']))
#     csv_writer.writerow([satid, summarization, captions, ground_image_ids])
#     print(f"Saved summarization for SatelliteID {satid}")

def save_summarization(satid, data, csv_writer):
    captions = "\n".join([f"{i+1}. {caption}" for i, caption in enumerate(data['captions'])])
    ground_image_ids = ", ".join(map(str, data['ground_image_ids']))
    csv_writer.writerow([satid, captions, ground_image_ids])
    print(f"Saved summarization for SatelliteID {satid}")

def main():
    torch.cuda.empty_cache()

    captions_file = 'captions/captions2.csv'
    satellite_centers_file = 'satellite_centers.pkl'
    satellite_to_captions = 'satellite_to_captions.pkl'
    output_dir = 'captions'
    output_file = os.path.join(output_dir, 'sat_captions.csv')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_over = False
    if not isfile(satellite_to_captions) or start_over:
        captions = read_captions(captions_file)
        satellite_centers = read_satellite_centers(satellite_centers_file)

        satellite_to_captions = {}
        for sat_index, ground_image_ids in enumerate(satellite_centers['ImageIds']):
            for ground_image_id in ground_image_ids:
                if ground_image_id in captions:
                    if sat_index not in satellite_to_captions:
                        satellite_to_captions[sat_index] = {'captions': [], 'ground_image_ids': []}
                    satellite_to_captions[sat_index]['captions'].append(captions[ground_image_id])
                    satellite_to_captions[sat_index]['ground_image_ids'].append(ground_image_id)

        with open('satellite_to_captions.pkl', 'wb') as file:
            pickle.dump(satellite_to_captions, file)
        
    with open('satellite_to_captions.pkl', 'rb') as file:
        satellite_to_captions = pickle.load(file)
    
    print("start captioning")
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['SatelliteID', 'Captions', 'GroundImageIDs'])

        # count = 0
        for satid, data in tqdm(satellite_to_captions.items(), desc="Processing satellite images"):
            # summarization = prompt_llama(data['captions'])
            # save_summarization(satid, summarization, data, csv_writer)
            save_summarization(satid, data, csv_writer)
            torch.cuda.empty_cache()  # Clear GPU cache after each summarization
            # count += 1
            # if count == 10:
            #     break
            
if __name__ == '__main__':
    main()

