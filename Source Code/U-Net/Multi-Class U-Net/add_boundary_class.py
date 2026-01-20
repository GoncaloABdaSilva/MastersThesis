import os
import cv2
import numpy as np



folder_pairs = [
    {
        'input' : 'train/gt',
        'output' : 'new_train/gt'
    },
    {
        'input' : 'test/gt',
        'output' : 'new_test/gt'
    },
    {
        'input' : 'val/gt',
        'output' : 'new_val/gt'
    }
]

# d=5, 5 + 1 + 5
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
for pair in folder_pairs:
        
    for filename in os.listdir(pair['input']):
        if not filename.lower().endswith('.png'):
            continue

        input_path = os.path.join(pair['input'], filename)
        output_path = os.path.join(pair['output'], filename)

        mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # Binarize to ensure only 0 or 255
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        dilated = cv2.dilate(binary, kernel, iterations=1)

        # New class = dilated - original
        new_class = cv2.subtract(dilated, binary)

        # Create 3-channel image
        output_mask = np.zeros((*binary.shape, 3), dtype=np.uint8)

        # Original gt: white [255,255,255]
        output_mask[binary == 255] = [255, 255, 255]

        # New class (dilation only): blue
        output_mask[new_class == 255] = [0, 0, 255]

        # Save result
        cv2.imwrite(output_path, output_mask)
        print(f"Saved: {output_path}")
