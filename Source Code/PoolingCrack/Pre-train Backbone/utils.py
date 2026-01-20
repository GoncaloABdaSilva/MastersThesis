import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from ImageNetValidationDS import ImageNetValidationDS
from ImageNetTrainDS import ImageNetTrainDS

best_val_loss = 1000.0 

def save_checkpoint(state, filename="pretrained_checkpoint.pth.tar"):
    print("\n=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer, scaler, scheduler):
    print("\n=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    scheduler.load_state_dict(checkpoint["lr_scheduler"])
    return checkpoint["curr_epoch"] + 1


def get_loaders(
    train_dir,
    val_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers,
    rank,
    world_size,
    pin_memory
):
    imagenet_train_dataset = ImageNetTrainDS(root_dir=train_dir, transform=train_transform)

    print(f"Created imagenet_train_dataset. {len(imagenet_train_dataset.class_to_idx)} classes created.")

    imagenet_val_dataset = ImageNetValidationDS(root_dir=val_dir, class_to_idx=imagenet_train_dataset.class_to_idx ,transform=val_transform)

    print("Created imagenet_val_dataset." )

    train_sampler = DistributedSampler(
        dataset= imagenet_train_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True
    )

    print("Created train_sampler")

    train_loader = DataLoader(
        dataset= imagenet_train_dataset, 
        batch_size= batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=train_sampler
    )

    print("Created train_loader")

    val_loader = DataLoader(
        dataset= imagenet_val_dataset,
        batch_size= batch_size,
        num_workers= num_workers,
        pin_memory= pin_memory
    )

    return train_loader, val_loader, train_sampler


def check_accuracy(loader, model, loss_fn, device):
    model.eval()

    global best_val_loss
    num_correct = 0
    total_loss = 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)
            num_correct += (preds == y).sum().item()

    avg_correct = 100 * num_correct / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset)

    print(f'''Validation accuracy: {avg_correct}%\n
          Validation loss: {avg_loss}''')

    model.train()

    if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        return avg_loss, True
    else:
        return avg_loss, False
