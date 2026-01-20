import os
import argparse
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import PoolingCrack
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    load_scores,
    #save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 5e-5
#DEVICE = "cuda" #if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
#NUM_EPOCHS = 200
NUM_WORKERS = 4
IMAGE_HEIGHT = 448  
IMAGE_WIDTH = 448
PIN_MEMORY = True
#IMAGE_NET_DIR = "/kaggle/input/ILSVRC"
CRACKTREE260_DS_KAGGLE = "/kaggle/input/gaps384"

TRAIN_IMG_DIR = CRACKTREE260_DS_KAGGLE + "/train/img"
TRAIN_MASK_DIR = CRACKTREE260_DS_KAGGLE + "/train/gt"
VAL_IMG_DIR = CRACKTREE260_DS_KAGGLE + "/val/img"
VAL_MASK_DIR = CRACKTREE260_DS_KAGGLE + "/val/gt"
PRE_TRAINED_CHECKPOINT = "/kaggle/input/poolingcrack/pretrained_checkpoint.pth.tar"

def dice_loss(logits, target, epsilon=1e-7):
    #print(f'LOGITS MAX: {torch.max(logits)}')
    #print(f'LOGITS MIN: {torch.min(logits)}')
    pred = torch.sigmoid(logits)

    pred_1d = torch.clamp(pred.view(-1), min = epsilon, max= 1 - epsilon)
    target_1d = target.view(-1)

    #print(pred_1d)
    '''numerator = 2 * torch.sum(pred * target)
    denominator = torch.sum(pred + target)'''

    numerator = (pred_1d * target_1d).sum()
    denominator = (pred_1d.sum() + target_1d.sum())
    dice_coef = (2 * (numerator + epsilon)) / (denominator + epsilon)

    return 1 - torch.mean(dice_coef)

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def train_fn(loader, model, optimizer, loss_fn, scaler, gpu_id):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop): #loop só funciona se todas as imagens num batch tiverem a mesma dimensão
        data = data.to(device=gpu_id, non_blocking=True)
        targets = targets.float().unsqueeze(1).to(device=gpu_id, non_blocking=True)

        # forward
        with torch.amp.autocast('cuda'):
            logits = model(data)
            loss = loss_fn(logits, targets)
        '''logits = model(data)
        loss = loss_fn(logits, targets)'''

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
            A.ColorJitter(p=0.5),
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
            A.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
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
    model = PoolingCrack(in_channels=3, out_channels=1).to(gpu_id)

    pretrained_checkpoint = torch.load(PRE_TRAINED_CHECKPOINT)
    pretrained_dict = pretrained_checkpoint["state_dict"]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()} #remove 'module.' prefix saved under DDP
    #Print for debug reasons
    missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)
    print(f"Loaded encoder weights. Missing keys: {len(missing_keys)} | Unexpected keys: {len(unexpected_keys)}")
    print(f"Received keys: {len(set(model.state_dict().keys()))}")

    #model_keys = set(model.state_dict().keys())
    pretrained_keys = set(pretrained_dict.keys())
    received_keys = pretrained_keys - set(unexpected_keys)

    for key in sorted(received_keys):
        print(f"    - {key}")


    model = DDP(model, device_ids=[gpu_id], output_device=gpu_id)
    model = model.float()

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

    '''loss_fn = lambda logits, targets: dice_loss(logits, targets)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    scaler = torch.amp.GradScaler('cuda')'''

    loss_fn = lambda logits, targets: dice_loss(logits, targets)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    scaler = torch.amp.GradScaler('cuda')

    #if LOAD_MODEL:
    if load_model:
        print(f"LOADING MODEL at GPU: {gpu_id}.")
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_checkpoint.pth.tar")
        next_epoch = load_checkpoint(torch.load(path), model.module, optimizer, scaler)
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
        train_sampler.set_epoch(epoch)

        if gpu_id == 0:
            print(f"\n[Epoch {epoch}]")
        
        train_fn(train_loader, model, optimizer, loss_fn, scaler, gpu_id)

        # save model
        checkpoint = {
            "state_dict": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "curr_epoch" : epoch,
            "scaler" : scaler.state_dict()
        }
        #if gpu_id == 0 or epoch == NUM_EPOCHS-1:

        check_accuracy(val_loader, model, checkpoint, loss_fn, device=gpu_id)
        
        if gpu_id ==0:
            save_checkpoint(checkpoint)
                    
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs, regardless of previous training.")
    parser.add_argument("--load_model", action=argparse.BooleanOptionalAction, required=True, help="Whether to load a pre-trained model (--load_model) or not (--no-load_model).")

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.epochs, args.load_model), nprocs=world_size)