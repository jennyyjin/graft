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
import pandas as pd
# from vllm import LLM, SamplingParams
from tqdm import tqdm
import os
# import torch
from os.path import isfile
# from rouge_score import rouge_scorer
print("finish import")

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

def combine_dataframes(test_file, train_file):
    """
    Combine the test and train CSV files and extract relevant columns.
    """
    test_df = pd.read_csv(test_file, header=None, names=['satellite_image', 'ground_image'])
    train_df = pd.read_csv(train_file, header=None, names=['satellite_image', 'ground_image'])
    combined_df = pd.concat([test_df, train_df], ignore_index=True)
    return combined_df


def save_summarization(satid, data, csv_writer):
    # print(data['captions'])
    captions = "\n".join([f"{i+1}. {caption}" for i, caption in enumerate(data['captions'])])
    # captions = []
    # for i, caption in enumerate(data['captions']):
    #     if "2. Describe the image. - " in caption:
    #         parts = caption.split('2. Describe the image. - ', 1)
    #         if len(parts) == 2:
    #             captions.append(f"{i+1}. {parts[1]}")
    #     elif "2. Describe the image. " in caption:
    #         parts = caption.split('2. Describe the image. ', 1)
    #         if len(parts) == 2:
    #             captions.append(f"{i+1}. {parts[1]}")
    #     elif "2. " in caption:
    #         parts = caption.split('2. ', 1)
    #         if len(parts) == 2:
    #             captions.append(f"{i+1}. {parts[1]}")

    # captions = "\n".join(captions)
    # print(captions)
    ground_image_ids = ", ".join(map(str, data['ground_image_ids']))
    csv_writer.writerow([satid, captions, ground_image_ids])
    # print(f"Saved summarization for SatelliteID {satid}")

def main():
    # torch.cuda.empty_cache()

    captions_file = 'captions/combined_captions2.csv'
    satellite_centers_file = 'satellite_centers_recentered.pkl'
    satellite_to_captions_dir = 'satellite_to_captions_160k.pkl'
    output_dir = 'captions'
    output_file = os.path.join(output_dir, 'sat_captions_160k.csv')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_over = False
    if not isfile(satellite_to_captions_dir) or start_over:
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

        with open(satellite_to_captions_dir, 'wb') as file:
            pickle.dump(satellite_to_captions, file)
        
    with open(satellite_to_captions_dir, 'rb') as file:
        satellite_to_captions = pickle.load(file)
    
    print("start captioning")
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['SatelliteID', 'Captions', 'GroundImageIDs'])

        # count = 0
        for satid, data in tqdm(satellite_to_captions.items(), desc="Processing satellite images"):
            # summarization = prompt_llama(data['captions'])
            # save_summarization(satid, summarization, data, csv_writer)
            try:
                save_summarization(satid, data, csv_writer)
            except:
                continue
            # torch.cuda.empty_cache()  # Clear GPU cache after each summarization
            # count += 1
            # if count == 10:
            #     break
            
if __name__ == '__main__':
    main()

# def main():
#     print('start')
#     # torch.cuda.empty_cache()
#     captions_file = 'captions/combined_captions.csv'
#     test_file = '/home/yx229/graft-data-collection/data/split_m2o_all_jp/test.csv'
#     train_file = '/home/yx229/graft-data-collection/data/split_m2o_all_jp/train.csv'
#     satellite_to_captions_file = 'satellite_to_captions_100k_recenter.pkl'
#     output_dir = 'captions'
#     output_file = os.path.join(output_dir, 'sat_captions_100k_recenter.csv')

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     start_over = False
#     if not os.path.isfile(satellite_to_captions_file) or start_over:
#         captions = read_captions(captions_file)
#         combined_df = combine_dataframes(test_file, train_file)

#         satellite_to_captions = {}
#         for _, row in tqdm(combined_df.iterrows(), desc="Associating captions", total=combined_df.shape[0]):
#             satellite_image_path = row['satellite_image']
#             ground_image_path = row['ground_image']
#             ground_image_id = extract_gid(ground_image_path)

#             if ground_image_id and int(ground_image_id) in captions:
#                 satid = satellite_image_path.split('/')[-1].split('_')[0]
#                 if satid not in satellite_to_captions:
#                     satellite_to_captions[satid] = {'captions': [], 'ground_image_ids': []}
#                 satellite_to_captions[satid]['captions'].append(captions[int(ground_image_id)])
#                 satellite_to_captions[satid]['ground_image_ids'].append(ground_image_id)

#         with open(satellite_to_captions_file, 'wb') as file:
#             pickle.dump(satellite_to_captions, file)

#     with open(satellite_to_captions_file, 'rb') as file:
#         satellite_to_captions = pickle.load(file)

#     with open(output_file, 'w', newline='') as csvfile:
#         csv_writer = csv.writer(csvfile)
#         csv_writer.writerow(['SatelliteID', 'Captions', 'GroundImageIDs'])

#         for satid, data in tqdm(satellite_to_captions.items(), desc="Processing satellite images"):
#             try:
#                 save_summarization(satid, data, csv_writer)
#             except Exception as e:
#                 print(f"Error processing SatelliteID {satid}: {e}")
#                 continue

# if __name__ == '__main__':
#     main()