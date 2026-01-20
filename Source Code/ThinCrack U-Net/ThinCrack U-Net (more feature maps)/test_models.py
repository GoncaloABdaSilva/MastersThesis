import os
import torch
import torchvision
from model import ThinCrack_UNET
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import CrackTree260
from train import (
    binary_focal_loss_split,
    #bce_dice_loss_function,
)

IMAGE_HEIGHT = 512  
IMAGE_WIDTH = 512
BATCH_SIZE = 10
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

def test_model(model, test_loader, filename, folder="test_saved_images/"):
    num_correct = 0
    num_pixels = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    total_loss = 0
    new_tp_dillated_y_1 = 0
    new_tp_dillated_y_2 = 0
    new_tp_dillated_y_5 = 0

    for idx, (x,y) in enumerate(test_loader):
        x = x.to(device="cuda")
        y = y.to(device="cuda")

        y = y.unsqueeze(1)

        with torch.no_grad():
            preds = model(x)
            #probs = torch.sigmoid(preds)
            probs = (preds >= 0.5).float()

            checkpoint_name = filename.replace('.pth.tar', '')
            subfolder_path = os.path.join(folder, checkpoint_name)

            if not os.path.exists(subfolder_path):
                os.mkdir(subfolder_path)

            torchvision.utils.save_image(
                probs, f"{subfolder_path}/pred_{idx}.png"
            )

            total_loss += loss_fn(preds, y).item() * x.size(0)

            num_correct += (probs == y).sum()
            true_positives += ((probs == y) & (y == 1)).sum()
            true_negatives += ((probs == y) & (y == 0)).sum()
            false_positives += ((probs != y) & (y == 0)).sum()
            false_negatives += ((probs != y) & (y == 1)).sum()
            num_pixels += torch.numel(probs)

            #Adjacent positives
            kernel_1 = torch.ones((1,1,3,3), device = y.device)
            dilated_y_1 = F.conv2d(y.float(), kernel_1, padding=1)
            dilated_y_1 = (dilated_y_1 > 0).float()
            new_tp_dillated_y_1 += ((probs != y) & (y == 0) & (dilated_y_1 == 1)).sum()

            kernel_2 = torch.ones((1,1,5,5), device = y.device)
            dilated_y_2 = F.conv2d(y.float(), kernel_2, padding=2)
            dilated_y_2 = (dilated_y_2 > 0).float()
            new_tp_dillated_y_2 += ((probs != y) & (y == 0) & (dilated_y_2 == 1)).sum()

            kernel_5 = torch.ones((1,1,11,11), device = y.device)
            dilated_y_5 = F.conv2d(y.float(), kernel_5, padding=5)
            dilated_y_5 = (dilated_y_5 > 0).float()
            new_tp_dillated_y_5 += ((probs != y) & (y == 0) & (dilated_y_5 == 1)).sum()


    avg_loss = total_loss / len(test_loader.dataset)

    print(
            f"Got {num_correct}/{num_pixels} correct pixels, with TP = {true_positives}, TN = {true_negatives}, FP = {false_positives}, FN = {false_negatives}."
            )
    curr_miou = ((true_positives/(true_positives + false_positives + false_negatives)) + (true_negatives/(true_negatives + false_negatives + false_positives)))/2
    curr_f1_score = (2 * true_positives)/(2 * true_positives + false_positives + false_negatives)

    print(
            f"""\nMetric scores: Accuracy = {(true_positives + true_negatives)/num_pixels}\n 
            Precision = {true_positives/(true_positives + false_positives)}\n 
            Recall = {true_positives/(true_positives + false_negatives)}\n 
            F1-Score = {curr_f1_score}\n 
            mIoU = {curr_miou}\n
            Testing Loss = {avg_loss}"""
        )
    
    print(f"\nDilated New True Positives: d=1 ={new_tp_dillated_y_1}, d=2 = {new_tp_dillated_y_2}, d=5 = {new_tp_dillated_y_5}")

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

    test_ds = CrackTree260(
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

    # Change according to correct loss function used
    loss_fn = lambda logits, targets: binary_focal_loss_split(logits, targets, alpha=3.0, gamma=1.0)

    for filename in os.listdir(checkpoints):
        if not filename.endswith(".pth.tar"):
            continue

        model_path = os.path.join(checkpoints, filename)
        checkpoint = torch.load(model_path, map_location='cuda:0')
        print(f"\n==> Testing {filename}")
        model = ThinCrack_UNET(in_channels=3, out_channels=1)
        model.load_state_dict(checkpoint["state_dict"])
        model.to('cuda:0')
        model.eval()
        test_model(model, test_loader, filename)

    

                



