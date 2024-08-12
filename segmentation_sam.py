from PIL import Image
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

image_path = "../graft-data-collection/data/jpimages/0007/00035109_N36.5971E137.8454_bbox.jpg"
image = Image.open(image_path)

image_array = np.array(image)

model_type = "vit_b" 
checkpoint_path = "sam_vit_b_01ec64.pth"
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image_array)

import matplotlib.pyplot as plt

plt.imshow(image)
for mask in masks:
    plt.imshow(mask['segmentation'], alpha=0.05)
plt.savefig("output.jpg")
