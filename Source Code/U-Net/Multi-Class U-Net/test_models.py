import os
import torch
from model import UNET
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from dataset import (
    ThreeClassCrackTree260,
    mask2rgb,
)

IMAGE_HEIGHT = 512  
IMAGE_WIDTH = 512
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True

def print_metrics(true_positives, false_positives, true_negatives, false_negatives, num_pixels):
    curr_miou = ((true_positives/(true_positives + false_positives + false_negatives)) + (true_negatives/(true_negatives + false_negatives + false_positives)))/2
    curr_f1_score = (2 * true_positives)/(2 * true_positives + false_positives + false_negatives)
    print(
            f"""\nMetric scores: Accuracy = {(true_positives + true_negatives)/num_pixels}\n 
            Precision = {true_positives/(true_positives + false_positives)}\n 
            Recall = {true_positives/(true_positives + false_negatives)}\n 
            F1-Score = {curr_f1_score}\n 
            mIoU = {curr_miou}"""
        )

def test_model(model, test_loader, filename, folder="test_saved_images/", epsilon=1e-7):
    num_correct = 0
    num_pixels = 0
    true_positives = [0, 0, 0]
    true_negatives = [0, 0, 0]
    false_positives = [0, 0, 0]
    false_negatives = [0, 0, 0]
    new_tp_dillated_y_5 = 0
    model.eval()

    for idx, (x,y) in enumerate(test_loader):
        x = x.to(device="cuda")
        y = y.to(device="cuda")

        y = y.long()

        with torch.no_grad():
            preds = model(x)
            probs = torch.argmax(preds, dim=1)

            checkpoint_name = filename.replace('.pth.tar', '')
            subfolder_path = os.path.join(folder, checkpoint_name)

            if not os.path.exists(subfolder_path):
                os.mkdir(subfolder_path)

            '''for i in range(probs.shape[0]):
                pred_np = probs[i].cpu().numpy().astype(np.uint8)
                rgb_img = mask2rgb(pred_np)
                Image.fromarray(rgb_img).save(
                    f"{subfolder_path}/pred_{idx}_{i}.png"
                )
            '''    

            num_pixels += torch.numel(probs)
            num_correct += (probs == y).sum()

            for cls in range(3):
                preds_cls = (probs == cls)
                targets_cls = (y == cls)
                
                true_positives[cls] += (preds_cls & targets_cls).sum().item()
                false_positives[cls] += (preds_cls & (~targets_cls)).sum().item()
                false_negatives[cls] += ((~preds_cls) & targets_cls).sum().item()
                true_negatives[cls] += ((~preds_cls) & (~targets_cls)).sum().item()

            target_crack = (y == 1).float().unsqueeze(1) 

            kernel_5 = torch.ones((1, 1, 11, 11), device=y.device)
            dilated_crack = (F.conv2d(target_crack, kernel_5, padding=5) > 0).float()
            pred_is_crack = (probs == 1).unsqueeze(1).float()

            gt_not_crack = (y != 1).unsqueeze(1).float()
            false_positive_near_crack = (pred_is_crack * gt_not_crack * dilated_crack).sum().item()
            new_tp_dillated_y_5 += int(false_positive_near_crack)
            
    print(f"\n\nGot {num_correct}/{num_pixels} correct pixels\n")
    '''for cls in range(3):
        if cls == 0:
            print("BACKGROUND PREDICTIONS:")
        elif cls == 1:
            print("CRACK PREDICTIONS:")
        else:
            print("CLOSE TO CRACK PREDICTIONS:")
    '''

    # Forge Crack and Crack Neighbourhood classes 
    total_tp = true_positives[1] +  new_tp_dillated_y_5 # mais falsepos[2] que sejam perto
    total_fp = false_positives[1] - new_tp_dillated_y_5
    total_tn = true_negatives[1] # Correct crack predictions
    total_fn = false_negatives[1] # Correct crack predictions

    print(f"TP = {total_tp}, TN = {total_tn}, FP = {total_fp}, FN = {total_fn}, FN converted to TP as per d=5 = {new_tp_dillated_y_5}"
    )

    print("Dillated predictions")
    curr_miou = ((total_tp/(total_tp + total_fp + total_fn)) + (total_tn/(total_tn + total_fn + total_fp)))/2
    curr_f1_score = (2 * total_tp)/(2 * total_tp + total_fp + total_fn)
    
    print(
        f"""\nMetric scores: Accuracy = {(total_tp + total_tn)/num_pixels}\n 
        Precision = {total_tp/(total_tp + total_fp + epsilon)}\n 
        Recall = {total_tp/(total_tp + total_fn + epsilon)}\n 
        F1-Score = {curr_f1_score}\n 
        mIoU = {curr_miou}"""
    )
    
    '''print(f"\nDilated New True Positives: d=1 ={new_tp_dillated_y_1}, d=2 = {new_tp_dillated_y_2}, d=5 = {new_tp_dillated_y_5}")

    print("\nIgnoring New True Positives.")
    print(f"\nd=1:")
    print_metrics(true_positives, false_positives - new_tp_dillated_y_1, true_negatives, false_negatives, num_pixels - new_tp_dillated_y_1)

    print(f"\nd=2:")
    print_metrics(true_positives, false_positives - new_tp_dillated_y_2, true_negatives, false_negatives, num_pixels - new_tp_dillated_y_2)

    print(f"\nd=5:")
    print_metrics(true_positives, false_positives - new_tp_dillated_y_5, true_negatives, false_negatives, num_pixels - new_tp_dillated_y_5)

    print(
        f"""Considering New True Positives as actual True Positives:
        """
    )

    print(f"\nd=1:")
    print_metrics(true_positives + new_tp_dillated_y_1, false_positives - new_tp_dillated_y_1, true_negatives, false_negatives, num_pixels)

    print(f"\nd=2:")
    print_metrics(true_positives + new_tp_dillated_y_2, false_positives - new_tp_dillated_y_2, true_negatives, false_negatives, num_pixels)

    print(f"\nd=5:")
    print_metrics(true_positives + new_tp_dillated_y_5, false_positives - new_tp_dillated_y_5, true_negatives, false_negatives, num_pixels)
'''

if __name__ == '__main__':
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

    directory = os.path.dirname(os.path.abspath(__file__))
    checkpoints = os.path.join(directory, "results")

    test_img_dir = os.path.join(directory, "test", "img").replace('\\', '/')
    test_gt_dir = os.path.join(directory, "test", "gt").replace('\\', '/') 

    test_transforms = A.Compose(
        [
            A.CenterCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
        ]
    )

    test_ds = ThreeClassCrackTree260(
        image_dir=test_img_dir,
        mask_dir=test_gt_dir,
        transform=test_transforms
    )

    test_loader = DataLoader(
        test_ds,
        batch_size= BATCH_SIZE,
        num_workers= NUM_WORKERS,
        pin_memory= PIN_MEMORY
    )

    weights = torch.tensor([1, 15, 5], dtype=torch.float32)
    
    #loss_fn = nn.CrossEntropyLoss(weight=weights)

    for filename in os.listdir(checkpoints):
        if not filename.endswith(".pth.tar"):
            continue

        model_path = os.path.join(checkpoints, filename)
        checkpoint = torch.load(model_path, map_location='cuda:0')
        print(f"\n==> Testing {filename}")
        model = UNET(in_channels=3, out_channels=3)
        model.load_state_dict(checkpoint["state_dict"])
        model.to('cuda:0')
        model.eval()
        test_model(model, test_loader, filename)
