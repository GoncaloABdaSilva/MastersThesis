import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class ImageNetTrainDS(Dataset):
    def __init__(self, root_dir ,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        self.class_to_idx = {}
        idx_counter = 0

        for class_name in sorted(os.listdir(root_dir)):
            self.class_to_idx[class_name] = idx_counter
            idx_counter += 1

            class_path = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = np.array(Image.open(img_path).convert("RGB"))
        
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.long)
