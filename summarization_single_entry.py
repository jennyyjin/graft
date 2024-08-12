import pandas as pd
import torch
import transformers
from transformers import LlamaForCausalLM, AutoTokenizer
from tqdm import tqdm
import csv
import os
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


# ground_captions = "1. The image captures a snowy mountain slope with a person skiing down the hill. The skier is wearing a green jacket and is positioned towards the right side of the slope. The snow-covered slope extends from the top left corner to the bottom right corner of the image."
# sat_caption = "The image is a satellite view of a large, grassy field with trees in the background. The field is situated next to a road, and there is a small hill in the middle of the field. The area appears to be a park or a recreational space, providing a green and open space for people to enjoy."
# messages = [
#     {"role": "system", "content": "You are a helpful AI assistant for summarizing image descriptions from the same region."},
#     {"role": "user", "content": f"The texts in the brackets are descriptions of images taken in the same region. Summarize these descriptions to provide an overview of the region. Be sure to capture any landscape properties or notable landmarks. Avoid unnecessary preambles but make reasonable assumptions. [{captions}] <Summary>:"}
# ]
ground_captions = "1. The image features a playground with a large wooden structure, likely a jungle gym, surrounded by a grassy area. The playground is situated on a hill, providing a picturesque view of the surrounding landscape. There are several playground equipment pieces, including a slide, a swing, and a couple of benches.\n2. The image features a group of people, including both adults and children, gathered around a playground. They are standing near a playground slide, with some of them holding their children's hands. The playground is located in a park, and there are several cars parked nearby.\n3. The image features a young child standing on a playground, wearing a gray shirt and black pants. The child appears to be barefoot, possibly enjoying the warm weather. The playground is equipped with various play structures, including a slide and a climbing structure.\n4. The image features a person riding a green, curved, and wavy slide, likely at a park or an amusement area. The person is sitting on the slide, enjoying the ride. The slide is surrounded by trees, adding a natural and serene atmosphere to the scene."
sat_caption = "The image is a satellite view of a large, well-maintained estate with a red brick building. The estate is surrounded by a lush green forest, providing a serene and picturesque setting. The property features a large circular driveway, which is likely used for parking or as an access point to the main building. The estate is situated near a highway, indicating that it is easily accessible by road."
# messages = [
#     {"role": "system", "content": "You are a helpful AI assistant for summarizing image descriptions from the same region."},
#     {"role": "user", "content": f"The texts in the brackets are descriptions of images taken in the same region. Summarize these descriptions to provide an overview of the region. Be sure to capture any landscape properties or notable landmarks. Avoid unnecessary preambles but make reasonable assumptions. [{captions}] <Summary>:"}
# ]
messages = [
    {"role": "system", "content": "You are a helpful AI assistant for summarizing image descriptions from the same region."},
    {"role": "user", "content": f"The texts in the brackets include one satellite image and ground images taken in the region enclosed by the satellite image. Summarize these descriptions to provide an overview of the region based on the information provided in both the satellite image and the ground images. Be sure to capture any landscape properties or notable landmarks. Avoid unnecessary preambles but make reasonable assumptions. \n\nSatellite image captions: [{sat_caption}] \n\nGround image captions: [{ground_captions}] \n\n <Summary>:"}
]
prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
sequences = pipeline(
    prompt,
    max_new_tokens=400,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)

result = ""

for seq in sequences:
    generated_text = seq['generated_text']
    if "<Summary>:" in generated_text:
        summarization = generated_text.split("<Summary>:")[1].strip()
        summarization = summarization.split("\n\n", 1)[-1].strip()  # Extract part after "assistant\n\n"
        result = summarization
    else:
        result = generated_text.strip()
        
        
print(result)

