import os
import sys
import argparse
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import UNET
#from model import ThinCrack_UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    load_scores,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
#LEARNING_RATE = 1e-4
#DEVICE = "cuda" #if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
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

'''def calculate_pos_weight(loader):
    total_pos = 0
    total_neg = 0
    for data, targets in tqdm(loader):
        targets = targets.view(-1)
        total_pos += targets.sum().item()
        total_neg += (1 - targets).sum().item()
    return total_neg / (total_pos + 1e-6) #to avoid cases where it is 0
'''
# This function will print all elements (pictures from dataset) id's, used by one gpu
# It is called two times, one per gpu, so we can manually compare them
# One rank (gpu_id) should have all even id's and the other all odd id's
'''
def verify_distributed_sampler(train_sampler, rank, epoch=0):
    train_sampler.set_epoch(epoch)
    indices = list(train_sampler)

    print(f"Rank {rank} indices: {indices}")  '''  


def binary_focal_loss_split(logits, targets, alpha, gamma):
    # Convert raw model outputs (logits) to probabilities in range [0, 1]
    probs = torch.sigmoid(logits)
    #probs = probs.clamp(min=1e-6, max=1-1e-6)

    # Compute the focal loss for positive class
    pos_mask = targets == 1  # Ground truth: crack pixel
    neg_mask = targets == 0  # Ground truth: non-crack pixel

    pos_loss = ( (1 - probs[pos_mask]) ** gamma ) * torch.log(probs[pos_mask])
    neg_loss = ( (probs[neg_mask]) ** gamma ) * torch.log(1 - probs[neg_mask])

    # Normalize by the sum of the modulating term (this is necessary for stability)
    pos_loss_sum = -1* (pos_loss.sum() / ((1 - probs[pos_mask]) ** gamma).sum())
    neg_loss_sum = -alpha * (neg_loss.sum() / (probs[neg_mask] ** gamma).sum())

    # Combine both losses
    '''
    false_positives_mask_6 = ((targets == 0) & (probs > 0.5) & (probs <= 0.6)).sum().item()
    print(f"False_positives under 0.6: {false_positives_mask_6}")

    false_positives_mask_7 = ((targets == 0) & (probs > 0.6) & (probs <= 0.7)).sum().item()
    print(f"False_positives under 0.7: {false_positives_mask_7}")

    false_positives_mask_8 = ((targets == 0) & (probs > 0.7) & (probs <= 0.8)).sum().item()
    print(f"False_positives under 0.8: {false_positives_mask_8}")

    false_positives_mask_9 = ((targets == 0) & (probs > 0.8) & (probs <= 0.9)).sum().item()
    print(f"False_positives under 0.9: {false_positives_mask_9}")

    false_positives_mask_above = ((targets == 0) & (probs > 0.9)).sum().item()
    print(f"False_positives over 0.9: {false_positives_mask_above}")

    false_positives_mask_one = ((targets == 0) & (probs == 1)).sum().item()
    print(f"False_positives equal to 1: {false_positives_mask_one}")
    '''

    loss = pos_loss_sum + neg_loss_sum
    return loss

def bce_dice_loss_function(logits, targets, epsilon=1e-7):
    probs = torch.sigmoid(logits)

    probs_1d = torch.clamp(probs.view(-1), min = epsilon, max= 1 - epsilon)
    targets_1d = targets.view(-1)

    intersection = (probs_1d * targets_1d).sum()

    #bce_loss = -(targets_1d * torch.log(probs_1d) + ((1 - targets_1d) * torch.log(1 - probs_1d)))
    #bce_loss = torch.nn.functional.binary_cross_entropy(probs_1d, targets_1d, reduction="mean")
    bce_loss = nn.BCEWithLogitsLoss()(logits, targets)

    dice_loss = 1 - (2 * intersection + epsilon)/(probs_1d.sum() + targets_1d.sum() + epsilon)

    return dice_loss + bce_loss

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def train_fn(loader, model, optimizer, loss_fn, scaler, gpu_id):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=gpu_id)
        targets = targets.float().unsqueeze(1).to(device=gpu_id)

        # forward
        with torch.amp.autocast('cuda'):
            logits = model(data)
            loss = loss_fn(logits, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main(rank: int, world_size: int, max_epochs: int, load_model: bool):
    
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
    
    ddp_setup(rank, world_size)
    gpu_id = rank
    print(f"GPU_ID/RANK: {gpu_id}")
    model = UNET(in_channels=3, out_channels=1).to(gpu_id)
    #model = ThinCrack_UNET(in_channels=3, out_channels=1).to(gpu_id)

    loss_fn = lambda logits, targets: binary_focal_loss_split(logits, targets, alpha=3.0, gamma=1.0)
    #loss_fn = lambda logits, targets: bce_dice_loss_function(logits, targets)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    model = DDP(model, device_ids=[gpu_id], output_device=gpu_id)

    train_loader, val_loader, train_sampler = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        rank,
        world_size,
        PIN_MEMORY,
    )

    #verify_distributed_sampler(train_sampler, rank)

    #weight = calculate_pos_weight(train_loader)
    #print(f"Weight: {weight}")
    #weight = 20
    #pos_weight = torch.FloatTensor ([weight]).to(gpu_id)
    #loss_fn = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
    #optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scaler = torch.amp.GradScaler('cuda')

    #if LOAD_MODEL:
    if load_model:
        print(f"LOADING MODEL at GPU: {gpu_id}.")
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_checkpoint.pth.tar")
        next_epoch = load_checkpoint(torch.load(path), model.module, optimizer, scaler, scheduler)
        load_scores()
        epochs = (next_epoch, max_epochs)
    else:
        epochs = max_epochs

    #for epoch in range(NUM_EPOCHS):
    epoch_range = range(0)

    if isinstance(epochs, int):
        epoch_range = range(epochs)
    else:
        epoch_range = range(*epochs)

    for epoch in epoch_range:
        if gpu_id == 0:
            print(f"\n[Epoch {epoch}]")
        
        train_sampler.set_epoch(epoch)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, gpu_id)

        # save model
        checkpoint = {
            "state_dict": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "curr_epoch" : epoch,
            "scaler" : scaler.state_dict(),
            "lr_scheduler": scheduler.state_dict()
        }
        #if gpu_id == 0 or epoch == NUM_EPOCHS-1:
        
        if gpu_id ==0 and epoch == max_epochs -1:
            save_checkpoint(checkpoint)
        
        #val_loss = check_accuracy(val_loader, model.module, checkpoint, loss_fn, device=gpu_id)
        val_loss = check_accuracy(val_loader, model, checkpoint, loss_fn, device=gpu_id)

        #if gpu_id == 0 and epoch == max_epochs -1:
        if epoch == max_epochs -1:
            save_predictions_as_imgs(val_loader, model, gpu_id)

        scheduler.step(val_loss)
            
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs, regardless of previous training.")
    parser.add_argument("--load_model", action=argparse.BooleanOptionalAction, required=True, help="Whether to load a pre-trained model (--load_model) or not (--no-load_model).")

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.epochs, args.load_model), nprocs=world_size)