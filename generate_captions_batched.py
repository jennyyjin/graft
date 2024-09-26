import os
import csv
from PIL import Image
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
import re
import gc
gc.collect()
torch.cuda.empty_cache()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

model_id = "llava-hf/llava-1.5-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(model_id, load_in_4bit=True)
processor = AutoProcessor.from_pretrained(model_id, pad_token="<pad>")

main_image_dir = "/scratch/datasets/bure/graft2/images"
output_file = "./captions_batched.csv"
batch_size = 8  

def image_paths(root_dir):
    image_paths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".jpg")):
                image_paths.append(os.path.join(subdir, file))
    return image_paths

def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image

def create_batches(image_paths, batch_size):
    for i in range(0, len(image_paths), batch_size):
        yield image_paths[i:i + batch_size]

def extract_description(output):
    description_match = re.search(r"ASSISTANT:(.*)", output, re.DOTALL)
    if description_match:
        return description_match.group(1).strip()
    return output.strip()

def generate_captions(image_batches, model, processor, csv_writer):
    for batch in image_batches:
        try:
            images = [load_image(image_path) for image_path in batch]
            prompts = ["USER: <image>\nDescribe the image.\nASSISTANT:"] * len(images)
            inputs = processor(prompts, images, return_tensors="pt").to("cuda", torch.float16)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_beams=4
                )

            outputs = processor.batch_decode(output_ids, skip_special_tokens=True)

            for image_path, output in zip(batch, outputs):
                caption_text = extract_description(output)
                csv_writer.writerow([image_path, caption_text])
                print(f"Processed {image_path}: {caption_text}")

            file.flush()
            
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing batch {batch}: {e}")

image_paths_list = image_paths(main_image_dir)
image_batches = create_batches(image_paths_list, batch_size)

with open(output_file, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(["Image Path", "Caption"])
    generate_captions(image_batches, model, processor, csv_writer)
