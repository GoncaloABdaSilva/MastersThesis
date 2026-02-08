import os
import random
from PIL import Image
import numpy as np

img_dir = "./croppedimg/croppedimg"
gt_dir = "./croppedgt/croppedgt"

output_sets = {
    './test': {'count': 44},
    './val': {'count': 100},
    './train': {}  # Remaining go here
}

# Process each .jpg file in img
for filename in os.listdir(img_dir):
    if not filename.lower().endswith(".jpg"):
        continue

    base_name = os.path.splitext(filename)[0]
    img_path = os.path.join(img_dir, base_name + ".jpg")
    gt_path = os.path.join(gt_dir, base_name + ".png")

    # Skip if corresponding .png ground truth doesn't exist
    if not os.path.exists(gt_path):
        print(f"Ground truth missing for: {filename}")
        continue

    img = Image.open(img_path)
    gt = Image.open(gt_path)

    # Changing ground truth from 1-bit pixels to 8-bit pixels
    if gt.mode == '1':
        gt = gt.convert('L')

    # Step 1: Rotate if needed
    if img.size == (540, 440):
        img_array = np.array(img)
        mean_color = img_array.mean(axis=(0, 1)).astype(np.uint8)
        padding = np.tile(mean_color, (8, 540, 1))
        padded_array = np.vstack((img_array, padding))
        img = Image.fromarray(padded_array)
        img.save(img_path)
    if gt.size == (540, 440):
        gt_array = np.array(gt)
        padding = np.zeros((8, 540), dtype=np.uint8)
        padded_array = np.vstack((gt_array, padding))
        gt = Image.fromarray(padded_array, mode='L')
        gt.save(gt_path)

    # Step 2: Random crop
    width, height = img.size
    left = random.randint(0, width - 448)
    upper = random.randint(0, height - 448)
    box = (left, upper, left + 448, upper + 448)

    img_cropped = img.crop(box)
    gt_cropped = gt.crop(box)

    # Step 3: Save both (overwrite)
    img_cropped.save(img_path)
    gt_cropped.save(gt_path)

print("Rotation and synchronized cropping complete.")
all_basenames = [
    os.path.splitext(f)[0]
    for f in os.listdir(img_dir)
    if f.lower().endswith(".jpg") and os.path.exists(os.path.join(gt_dir, os.path.splitext(f)[0] + ".png"))
]

print(f'{len(all_basenames)} total basenames.')

# Shuffle for randomness
random.shuffle(all_basenames)

# Assign images to each set based on counts
used = set()
start = 0
for set_name, info in output_sets.items():
    count = info.get('count', None)
    if count is not None:
        output_sets[set_name]['files'] = all_basenames[start:start+count]
        used.update(output_sets[set_name]['files'])
        start += count

# Remaining images go to train
remaining = list(set(all_basenames) - used)
output_sets['./train']['files'] = remaining

# Copy images and GTs to their folders
for set_name, info in output_sets.items():
    for base in info['files']:
        img_path = os.path.join(img_dir, base + ".jpg")
        gt_path = os.path.join(gt_dir, base + ".png")

        img = Image.open(img_path)
        gt = Image.open(gt_path)

        img.save(os.path.join(set_name, "img", base + ".jpg"))
        gt.save(os.path.join(set_name, "gt", base + ".png"))

    print(f"{set_name}: {len(info['files'])} images saved.")

print("Dataset split and save complete.")
