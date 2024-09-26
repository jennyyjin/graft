import os
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, AutoTokenizer
import matplotlib.pyplot as plt
import random
import shutil
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from shutil import copy
import torch.nn as nn
import requests
import urllib.request
from IPython.display import display, clear_output
import io
import random

class GRAFT(nn.Module):
    def __init__(self, CLIP_version="openai/clip-vit-base-patch16", temp=False, bias_projector=True):
        super().__init__()
        self.satellite_image_backbone = CLIPVisionModelWithProjection.from_pretrained(CLIP_version)
        self.patch_size = self.satellite_image_backbone.config.patch_size

        self.projector = nn.Sequential(
            nn.LayerNorm(self.satellite_image_backbone.config.hidden_size, eps=self.satellite_image_backbone.config.layer_norm_eps),
            nn.Linear(self.satellite_image_backbone.config.hidden_size, self.satellite_image_backbone.config.projection_dim, bias=bias_projector),
        )
        self.patch_size = self.satellite_image_backbone.config.patch_size
        self.norm_dim = -1

        self.temp = temp
        if temp:
            self.register_buffer("logit_scale", torch.ones([]) * (1 / 0.07))

    def forward(self, image_tensor):
        hidden_state = self.satellite_image_backbone(image_tensor).last_hidden_state
        satellite_image_features = F.normalize(self.projector(hidden_state), dim=self.norm_dim)
        return satellite_image_features

    def forward_features(self, image_tensor):
        embed = self.satellite_image_backbone(image_tensor).image_embeds
        satellite_image_features = F.normalize(embed)
        return satellite_image_features

device = "cuda"
model = GRAFT(temp=True, bias_projector=False).to(device)

# load graft checkpoint
ckpt_path = "graft_weights/2024-08-05-14-18-55_CLIP_Contrastive_M2O_Cross_Image_Only_wd_1e-2_lr_1e-5_fixed_normalization_fixed_projector_initialization/checkpoints/last.ckpt"
state_dict = torch.load(ckpt_path)
model.load_state_dict(state_dict['state_dict'], strict=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

textmodel = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch16").eval().to(device)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

def get_features(image_path):
    image_tensor = load_image(image_path)
    with torch.no_grad():
        image_features = model.forward_features(image_tensor)
    return image_features

def search(query, image_folder):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        text_features = textmodel(**inputs).text_embeds

    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    similarities = {}
    for path in image_paths:
        image_features = get_features(path)
        sim = torch.cosine_similarity(text_features, image_features, dim=-1).item()
        similarities[path] = sim

    sorted_paths = sorted(similarities, key=similarities.get, reverse=True)
    return sorted_paths

def save_images(top_images, des_folder):
    os.makedirs(des_folder, exist_ok=True)
    for image_path in top_images:
        filename = os.path.basename(image_path)
        des_path = os.path.join(des_folder, filename)
        shutil.copy(image_path, des_path)

query = "railroad"
image_folder = "/scratch/datasets/bure/eval/jpimages_osm_eval/japan_images"
top_images = search(query, image_folder)

des_folder = 'outputs/text_retrieval/'

# save top 5 images
save_images(top_images[:5], des_folder)
