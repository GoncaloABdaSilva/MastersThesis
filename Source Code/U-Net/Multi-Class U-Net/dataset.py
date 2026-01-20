import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

LABEL_TO_COLOR = {
    0: [0,0,0],
    1: [255,255,255],
    2: [255,0,0]
}

def rgb2mask(rgb_gt):
    mask = np.zeros((rgb_gt.shape[0], rgb_gt.shape[1]), dtype=np.uint8)
    for k,v in LABEL_TO_COLOR.items():
        mask[np.all(rgb_gt == v, axis=2)] = k
    return mask

def mask2rgb(triple_mask):
    rgb = np.zeros(triple_mask.shape+(3,), dtype=np.uint8)
    
    for i in np.unique(triple_mask):
        rgb[triple_mask==i] = LABEL_TO_COLOR[i]
            
    return rgb

class ThreeClassCrackTree260(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        #mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        #mask[mask == 255.0] = 1.0
        mask = rgb2mask(mask_rgb)
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
          
        return image, mask
