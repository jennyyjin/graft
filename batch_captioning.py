import os
from PIL import Image
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from tqdm import tqdm
import gc
import pickle
import pandas as pd
import csv

def initialize_llava():
    """Initialize LLava model."""
    model_id = "llava-hf/llava-1.5-7b-hf"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    caption_model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    caption_processor = AutoProcessor.from_pretrained(model_id, pad_token="<pad>")
    return caption_model, caption_processor

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def generate_captions(model, processor, images, prompts):
    inputs = processor(prompts, images, return_tensors="pt").to("cuda", torch.float16)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=512, temperature=0.7, top_p=0.9, num_beams=4
        )
    captions = processor.batch_decode(outputs, skip_special_tokens=True)
    parsed_captions = [caption.split("ASSISTANT:", 1)[-1].strip() for caption in captions]
    return parsed_captions


def get_all_satellite_images(satellite_image_dir):
    """
    Traverse the nested folder structure and collect all satellite image paths.
    """
    satellite_files = []
    for root, _, files in os.walk(satellite_image_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):  # Add valid extensions
                satellite_files.append(os.path.join(root, file))
    return satellite_files

def process_satellite_images(satellite_image_dir, ground_image_dir, sat_centers, output_file, caption_model, caption_processor):
    """Process satellite images and avoid redundancy by tracking already processed images."""
    satellite_files = get_all_satellite_images(satellite_image_dir)

    # Track already processed satellite images
    processed_images = set()
    if os.path.exists(output_file):
        with open(output_file, mode="r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            processed_images = {row["Satellite Image"] for row in reader}

    # Open the CSV file for writing rows in real time
    with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Satellite Image", "Ground Images", "Ground Image IDs", "Satellite Caption", "Ground Captions"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for satellite_image in tqdm(satellite_files, desc="Processing satellite images"):
            # Load satellite image
            satellite_path = os.path.join(satellite_image_dir, satellite_image)
            satellite_img = load_image(satellite_path)
            satellite_image_id = int(os.path.splitext(os.path.basename(satellite_image))[0])

            # Get corresponding ground image IDs from the pickle file
            ground_image_ids = sat_centers['ImageIds'][satellite_image_id]

            # Find ground images based on folder structure
            ground_image_paths = []
            for ground_image_id in ground_image_ids:
                ground_image_folder = os.path.join(
                    ground_image_dir,
                    f"{ground_image_id // 1000 % 10}/{ground_image_id // 100 % 10}/{ground_image_id // 10 % 10}/{ground_image_id % 10}"
                )
                ground_image_file = f"{ground_image_id}.jpg"  # Assuming images are saved with .jpg extension
                ground_image_path = os.path.join(ground_image_folder, ground_image_file)
                if os.path.exists(ground_image_path):
                    ground_image_paths.append(ground_image_path)

            # Generate captions for satellite and ground images
            satellite_caption = generate_captions(caption_model, caption_processor, [satellite_img], [
                "USER: <image>\nProvide a detailed description of the image, focusing strictly on its visual elements. Avoid inferring or mentioning anything not directly visible.\nASSISTANT:"
            ])[0]
            ground_imgs = [load_image(img) for img in ground_image_paths]
            ground_prompts = [
                "USER: <image>\nProvide a detailed description of the image, focusing strictly on its visual elements. Avoid inferring or mentioning anything not directly visible.\nASSISTANT:"
            ] * len(ground_imgs)
            ground_captions = generate_captions(caption_model, caption_processor, ground_imgs, ground_prompts)

            # Write the row to the CSV in real time
            writer.writerow({
                "Satellite Image": satellite_image,
                "Ground Images": ", ".join(ground_image_paths),
                "Ground Image IDs": ", ".join(map(str, ground_image_ids)),
                "Satellite Caption": satellite_caption,
                "Ground Captions": ", ".join(ground_captions)
            })
            csvfile.flush()

if __name__ == "__main__":
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    import argparse

    parser = argparse.ArgumentParser(description="Process satellite images for captioning.")
    parser.add_argument("--data_path_root", type=str, required=True, help="Root path to the dataset.")
    args = parser.parse_args()

    data_path_root = args.data_path_root
        satellite_image_dir = os.path.join(data_path_root, "naipimages")
        ground_image_dir = os.path.join(data_path_root, "images")
        sat_centers_path = os.path.join(data_path_root, "src", "satellite_centers.pkl")
    output_file = "./caption_results.csv"

    # Initialize LLava model
    caption_model, caption_processor = initialize_llava()

    # Load satellite centers
    with open(sat_centers_path, 'rb') as f:
        sat_centers = pickle.load(f)

    # Process satellite images
    process_satellite_images(
        satellite_image_dir,
        ground_image_dir,
        sat_centers,
        output_file,
        caption_model,
        caption_processor
    )

    print(f"Results saved to {output_file}")
