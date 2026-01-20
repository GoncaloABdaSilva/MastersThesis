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
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import PoolingCrackEncoder
from utils import (
    save_checkpoint,
    get_loaders,
    check_accuracy,
    load_checkpoint,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
#DEVICE = "cuda" #if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 130
NUM_WORKERS = 2
ACCUMULATION_STEPS = 4
IMAGE_HEIGHT = 224  
IMAGE_WIDTH = 224
PIN_MEMORY = True
IMAGE_NET_MINI_DIR = "/kaggle/input/imagenetmini-1000/imagenet-mini"
#CRACKTREE260_DS_KAGGLE = "/kaggle/input/resampled-cracktree260/CrackTree260"

#TRAIN_IMG_DIR = "/kaggle/input/5-class-imagenet-will-delete"
TRAIN_IMG_DIR = IMAGE_NET_MINI_DIR + "/train"
VAL_IMG_DIR = IMAGE_NET_MINI_DIR + "/val"
#VAL_ANNOTATION_DIR = IMAGE_NET_DIR + "/Annotations/CLS-LOC/val" 
#TRAIN_TXT_FILE = IMAGE_NET_DIR + "/ImageSets/CLS-LOC/train_cls.txt"
#TRAIN_TXT_FILE = "/kaggle/input/pre-train-poolingcrack-encoder/format.txt"

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def train_fn(loader, model, optimizer, scaler, gpu_id):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop): #loop só funciona se todas as imagens num batch tiverem a mesma dimensão
        data = data.to(device=gpu_id, non_blocking=True)
        labels = targets.to(device=gpu_id, non_blocking=True)

        # forward
        with torch.amp.autocast('cuda'):
            logits = model(data)
            loss = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, labels)

        # backward
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main(rank: int, world_size: int, max_epochs: int, load_model: bool):
    
    train_transform = A.Compose(
        [
            A.RandomResizedCrop(size=[IMAGE_HEIGHT, IMAGE_WIDTH]),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=256, width=256),
            A.CenterCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ],
    )

    ddp_setup(rank, world_size)
    gpu_id = rank
    print(f"GPU_ID/RANK: {gpu_id}")
    model = PoolingCrackEncoder(in_channels=3, out_channels=1000).to(gpu_id)
    model = DDP(model, device_ids=[gpu_id], output_device=gpu_id)
    model = model.float()

    print("Model's DDP has been created")

    train_loader, val_loader, train_sampler = get_loaders(
        TRAIN_IMG_DIR,
        VAL_IMG_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        rank,
        world_size,
        PIN_MEMORY,
    )

    print("get_loaders() completed")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=200)
    scaler = torch.amp.GradScaler('cuda')

    #if LOAD_MODEL:
    if load_model:
        print(f"LOADING MODEL at GPU: {gpu_id}.")
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_checkpoint.pth.tar")
        next_epoch = load_checkpoint(torch.load(path), model.module, optimizer, scaler, scheduler)
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
        train_sampler.set_epoch(epoch)

        if gpu_id == 0:
            print(f"\n[Epoch {epoch}]")
        
        train_fn(train_loader, model, optimizer, scaler, gpu_id)

        # save model
        checkpoint = {
            "state_dict": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "curr_epoch" : epoch,
            "scaler" : scaler.state_dict(),
            "lr_scheduler": scheduler.state_dict()
        }

        val_loss, isBestValidationLoss = check_accuracy(val_loader, model, nn.CrossEntropyLoss(label_smoothing=0.1), device=gpu_id)
        
        if gpu_id == 0:
            if epoch == max_epochs-1:
                save_checkpoint(checkpoint)

            if isBestValidationLoss:
                save_checkpoint(checkpoint, "pretrained_best_loss_checkpoint.pth.tar")

        scheduler.step()
                    
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs, regardless of previous training.")
    parser.add_argument("--load_model", action=argparse.BooleanOptionalAction, required=True, help="Whether to load a pre-trained model (--load_model) or not (--no-load_model).")

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.epochs, args.load_model), nprocs=world_size)