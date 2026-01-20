import os
import sys
import argparse
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn

DEVICE = 'cuda:0'

#torch.load = functools.partial(torch.load, weights_only=False)

os.chdir("/kaggle/working")  # NOT inside sam2 repo

# Add the inner sam2 package folder to path
sys.path.append("/kaggle/working/sam2")

import sam2.build_sam as build_sam

from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir
GlobalHydra.instance().clear()

# Initialize Hydra explicitly pointing to the configs folder
initialize_config_dir(config_dir="/kaggle/working/sam2/sam2/configs", version_base=None)

def patched_load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    #if "model" in checkpoint:
    state_dict = checkpoint["model"]
    #else:
        #state_dict = checkpoint
    model.load_state_dict(state_dict)
    print(f"[Patched] Loaded checkpoint from {ckpt_path}")

build_sam._load_checkpoint = patched_load_checkpoint

sam2 = build_sam.build_sam2(
   "sam2.1/sam2.1_hiera_s.yaml", 
   "/kaggle/working/sam2.1_hiera_small.pt",
    device= DEVICE
)

os.chdir("/kaggle/input/sam2-fine-tuning-code") 

from sam2_seg_wrapper import SAM2SegWrapper


from utils import (
    save_checkpoint,
    get_loaders,
    check_accuracy,
    create_json_scores_file,
    get_test_loaders,
    test_checkpoint
)

# Hyperparameters etc.
#LEARNING_RATE = 1e-4
#DEVICE = "cuda" #if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
#NUM_EPOCHS = 100
NUM_WORKERS = 2 # 4 CPU cores for 2 GPUs, so 2 CPU Cores per GPU
IMAGE_HEIGHT = 512  
IMAGE_WIDTH = 512
PIN_MEMORY = True
#LOAD_MODEL = False
CRACKTREE260_DS_KAGGLE = "/kaggle/input/resampled-cracktree260/CrackTree260"
TRAIN_IMG_DIR = CRACKTREE260_DS_KAGGLE + "/train/img"
TRAIN_MASK_DIR = CRACKTREE260_DS_KAGGLE + "/train/gt"
VAL_IMG_DIR = CRACKTREE260_DS_KAGGLE + "/val/img"
VAL_MASK_DIR = CRACKTREE260_DS_KAGGLE + "/val/gt"

CRACKTREE260_TEST_DS_KAGGLE = "/kaggle/input/resampled-test-cracktree260"
TEST_IMG_DIR = CRACKTREE260_TEST_DS_KAGGLE + "/test_cracktree260/img"
TEST_MASK_DIR = CRACKTREE260_TEST_DS_KAGGLE + "/test_cracktree260/gt"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.amp.autocast('cuda'):
            logits = model(data)
            loss = loss_fn(logits, targets)

        # backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main(epochs: int):
    
    train_transform = A.Compose(
        [
            A.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussianBlur(blur_limit=(11, 11), sigma_limit=(0.1, 3.0), p=0.5),
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=50, p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.CenterCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    #predictor = SAM2ImagePredictor(sam2)
    model = SAM2SegWrapper(sam2, IMAGE_HEIGHT, IMAGE_WIDTH)

    model = model.to(DEVICE)
    #sam2.model.sam_mask_decoder.train(True)
    
    #model = SAM2SegWrapper(sam2, IMAGE_HEIGHT, IMAGE_WIDTH)
    #model = model.to(DEVICE)

    weight = 20
    pos_weight = torch.FloatTensor([weight]).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight = pos_weight)

    optimizer=torch.optim.AdamW(
        params= filter (lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, 
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    scaler = torch.amp.GradScaler()

    create_json_scores_file()

    for epoch in range(epochs):
        
        print(f"\n[Epoch {epoch}]")
        
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "curr_epoch" : epoch,
            "scaler" : scaler.state_dict(),
            "lr_scheduler": scheduler.state_dict()
        }
        #if gpu_id == 0 and epoch == epochs -1:
        
        val_loss = check_accuracy(val_loader, model, checkpoint, loss_fn, DEVICE)

        if epoch == epochs -1:
            save_checkpoint(checkpoint)

        scheduler.step(val_loss)
            
def test():
    checkpoint_files = [f for f in os.listdir("/kaggle/working") if f.endswith(".pth.tar")]

    sam2_model = build_sam.build_sam2(
        "sam2.1/sam2.1_hiera_s.yaml",
        "/kaggle/working/sam2.1_hiera_small.pt",
        device=DEVICE
    )

    model_wrapper = SAM2SegWrapper(sam2_model).to(DEVICE)

    weight = 20
    pos_weight = torch.FloatTensor([weight]).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight = pos_weight)

    test_transforms = A.Compose(
        [
            A.CenterCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    test_loader = get_test_loaders(
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        test_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    for ckpt_file in checkpoint_files:
        ckpt_path = os.path.join("/kaggle/working", ckpt_file)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model = model_wrapper
        model.load_state_dict(ckpt["state_dict"], strict=False)

        print(f"\n==> Testing {os.path.basename(ckpt_path)}")
        test_checkpoint(test_loader, model, loss_fn, DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs, regardless of previous training.")
    #parser.add_argument("--load_model", action=argparse.BooleanOptionalAction, required=True, help="Whether to load a pre-trained model (--load_model) or not (--no-load_model).")

    args = parser.parse_args()
    #mp.spawn(main, args=(world_size, args.epochs, args.load_model), nprocs=world_size)
    #mp.spawn(main, args=(world_size, args.epochs), nprocs=world_size)
    main(args.epochs)

    test()