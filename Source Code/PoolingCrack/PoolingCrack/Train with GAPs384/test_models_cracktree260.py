import os
import torch
import torchvision
from model import PoolingCrack
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import Gaps384
from train import dice_loss

IMAGE_HEIGHT = 512  
IMAGE_WIDTH = 512
BATCH_SIZE = 1
NUM_WORKERS = 4
PIN_MEMORY = True

def test_model(model, test_loader, filename, folder="test_cracktree260_saved_images/"):
    num_correct = 0
    num_pixels = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    total_loss = 0

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

if __name__ == '__main__':
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

    directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(directory)
    checkpoints = os.path.join(directory, "results")

    test_img_dir = os.path.join(parent_directory, "test_cracktree260", "img").replace('\\', '/')
    test_gt_dir = os.path.join(parent_directory, "test_cracktree260", "gt").replace('\\', '/') 

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

    test_ds = Gaps384(
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
    loss_fn = lambda logits, targets: dice_loss(logits, targets)

    for filename in os.listdir(checkpoints):
        if not filename.endswith(".pth.tar"):
            continue

        model_path = os.path.join(checkpoints, filename)
        checkpoint = torch.load(model_path, map_location='cuda:0')
        print(f"\n==> Testing {filename}")
        #model = PoolingCrack(in_channels=3, out_channels=1)

        with torch.no_grad():  # Prevent unwanted gradient tracking
            model = PoolingCrack(in_channels=3, out_channels=1)

        state_dict = checkpoint["state_dict"]

        for k, v in state_dict.items():
            if isinstance(v, torch.nn.Parameter):
                state_dict[k] = v.data

        model.load_state_dict(checkpoint["state_dict"])
        model.to('cuda:0')
        model.eval()
        test_model(model, test_loader, filename)
