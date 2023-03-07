import torch
import os
import glob
import gc
import random

torch.cuda.empty_cache()
gc.collect()

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='local')

# Images
dir = "/disk/scratch_big/data/input/datasets/test/images/"

for i in range(0,11):
    imgs = glob.glob(f"{dir}*.jpg")[i*100:(i+1)*100]
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print(f'Inferencing {i*100}~{(i+1)*100}')
    # Inference
    results = model(imgs)
    results.print()  # or .show(), .save()

imgs = random.sample(glob.glob(f"{dir}*.jpg"),1)
# Inference
results = model(imgs)
results.print()  # or .show(), .save()