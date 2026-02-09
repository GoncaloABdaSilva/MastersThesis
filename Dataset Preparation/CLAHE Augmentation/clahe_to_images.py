import cv2
import numpy as np
import os

clipLimit_H = 5
clipLimit_S = 40
clipLimit_V = 5
tileGridSize = (8,8)

def apply_clahe(image):
    rgb_clahe = []

    for i in range(3):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tileGridSize) #paper does not specify clipLimit
        rgb_clahe.append(clahe.apply(image[:,:,i]))

    rgb_clahe = cv2.merge(rgb_clahe) #merge rgb channels


    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    h_clahe = cv2.createCLAHE(clipLimit=clipLimit_H, tileGridSize=tileGridSize).apply(h)
    s_clahe = cv2.createCLAHE(clipLimit=clipLimit_S, tileGridSize=tileGridSize).apply(s)
    v_clahe = cv2.createCLAHE(clipLimit=clipLimit_V, tileGridSize=tileGridSize).apply(v)

    hsv_clahe = cv2.merge([h_clahe, s_clahe, v_clahe])
    hsv_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

    final_enhanced = cv2.addWeighted(rgb_clahe, 0.5, hsv_clahe, 0.5, 0)

    return final_enhanced

    

if __name__ == "__main__":
    #input_folder = "test_cracktree260/img"
    input_folder = "images_with_shadows_clahe/img"


    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)
        
        final_img = apply_clahe(image)

        cv2.imwrite(img_path, final_img)
        print(f"Replaced: {filename}")

    print("Done :)")