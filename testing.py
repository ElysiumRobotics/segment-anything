# Testing 1

import cv2
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry # Run after cv2
import matplotlib.pyplot as plt
import numpy as np

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

file = "/home/Projects/uFibers/Elysium_Micrographs/2023/spools/spool_224/20230426/bundleset_50_1/1/cross-section/230427_1246_000_G001_I.jpg"
image = cv2.imread(file)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize image so that GPU can handle it. 
# - This should check to see if the original is too big and then scale it.
scale_percent = 20 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
print('Original Dimensions : ',image.shape)
print('Resized Dimensions : ',resized.shape)


sam = sam_model_registry["vit_h"](checkpoint="model/sam_vit_h_4b8939.pth")
device = "cuda"
sam.to(device=device)

#predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,
    )
masks = mask_generator.generate(resized)

min_area = 350
max_area = 7500
masks_filtered = [ m for m in masks if (m["area"] > min_area) and (m["area"] < max_area) ]

areas = [ m["area"] for m in masks_filtered ]
counts, bins = np.histogram(areas, bins=100)
plt.hist(bins[:-1], bins, weights=counts)


plt.figure(figsize=(20,20))
plt.imshow(resized)
show_anns(masks_filtered)
plt.axis('off')
plt.show() 

print("pausing here")



