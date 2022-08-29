import json
from tqdm import tqdm 
import numpy as np

a = json.load(open("data/sunrefer/train_sunrefer.json"))
b = json.load(open("data/sunrefer/val_sunrefer.json"))
dataset = a + b

XMAX, YMAX, ZMAX = -10000, -10000, -10000
XMIN, YMIN, ZMIN = 10000, 10000, 10000
for data in tqdm(dataset):
    path = data['point_cloud_path']
    points = np.load(data['point_cloud_path'])['pc'][:, :3]
    target = np.array(data['object_box'], dtype=np.float32)
    xmax, ymax, zmax = points.max(axis=0)
    xmin, ymin, zmin = points.min(axis=0)
    XMAX = max(XMAX, xmax)
    YMAX = max(YMAX, ymax)
    ZMAX = max(ZMAX, zmax)
    XMIN = min(XMIN, xmin)
    YMIN = min(YMIN, ymin)
    ZMIN = min(ZMIN, zmin)
    
#     print(target)
#     break
print(XMAX, YMAX, ZMAX, XMIN, YMIN, ZMIN)