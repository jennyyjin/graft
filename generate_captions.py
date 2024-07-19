import os
import csv
from PIL import Image
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
import re

model_path = "liuhaotian/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

main_image_dir = "../graft/data/images_processed"
output_file = "./caption.csv"

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

def generate_captions(image_paths, model, image_processor, tokenizer, csv_writer):
    for image_path in image_paths:
        try:
            image = load_image(image_path)
            images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

            prompt = "Describe the image."
            qs = prompt
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in qs:
                if model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv_mode = "llava_v0"
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .to(model.device)
            )

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_beams=5,
                    max_new_tokens=512,
                    use_cache=True,
                )

            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            if output is not None:
                caption_text = output.strip()
            else:
                caption_text = "No caption generated"

            csv_writer.writerow([image_path, caption_text])
            file.flush()
            print(f"Processed {image_path}: {caption_text}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

image_paths = image_paths(main_image_dir)

with open(output_file, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(["Image Path", "Caption"])
    generate_captions(image_paths, model, image_processor, tokenizer, csv_writer)
