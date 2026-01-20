import torch
import json
import os
import torchvision
from dataset import CrackTree260
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

best_f1_score = 0
best_miou = 0
best_loss = 4.0
avg_loss_list = []

def create_json_scores_file():
    data = { "best_f1_score" : 0.0,
        "best_miou" : 0.0, 
        "best_loss" : 4.0,
        "avg_loss_list": [],}
    
    with open("/kaggle/working/scores.json", 'w') as jfile:
        json.dump(data, jfile)

def load_scores():
    global best_f1_score, best_miou, best_loss, avg_loss_list
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scores.json")
    with open(json_path, 'r') as jfile:
        data = json.load(jfile)
        best_f1_score = data["best_f1_score"]
        best_miou = data["best_miou"]
        best_loss = data["best_loss"]
        avg_loss_list = data["avg_loss_list"]

def save_json_score():
    #.item() as they are torch.Tensor
    data = { "best_f1_score" : best_f1_score, 
        "best_miou" : best_miou, 
        "best_loss": best_loss,
        "avg_loss_list": avg_loss_list,}
    
    with open("scores.json", 'w') as new_jfile:
            json.dump(data, new_jfile)
    
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("\n=> Saving checkpoint")
    torch.save(state, filename)

def save_best_f1_score(state, filename="best_f1_score.pth.tar"):
    print("=> Saving best f1_score checkpoint.")
    torch.save(state, filename)
    #save_json_score()

def save_best_miou(state, filename="best_miou.pth.tar"):
    print("=> Saving best miou checkpoint.")
    torch.save(state, filename)    
    #save_json_score()

def save_best_loss(state, filename="best_loss.pth.tar"):
    print("=> Saving best loss checkpoint.")
    torch.save(state, filename)
    #save_json_score()


def load_checkpoint(checkpoint, model, optimizer, scaler, scheduler):
    print("\n=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    scheduler.load_state_dict(checkpoint["lr_scheduler"])
    return checkpoint["curr_epoch"] + 1

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers,
    rank,
    world_size,
    pin_memory
):
    train_ds = CrackTree260(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_sampler = DistributedSampler(
        train_ds, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=train_sampler
    )

    val_ds = CrackTree260(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    '''val_sampler = DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank, 
        shuffle=False
    )'''

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory#,
        #sampler=val_sampler

    )

    return train_loader, val_loader, train_sampler

def check_accuracy(loader, model, checkpoint, loss_fn, device):
    global best_f1_score, best_miou, best_loss, avg_loss_list

    num_correct = 0
    num_pixels = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    total_loss = 0
    model.eval() # Set the module in evaluation mode

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            #preds = torch.sigmoid(model(x))
            probs = model(x)
            preds = torch.sigmoid(probs)
            preds = (preds > 0.5).float()

            #total_loss += loss_fn(probs, y).item()  
            total_loss += loss_fn(probs, y).item() * x.size(0)

            num_correct += (preds == y).sum()
            true_positives += ((preds == y) & (y == 1)).sum()
            true_negatives += ((preds == y) & (y == 0)).sum()
            false_positives += ((preds != y) & (y == 0)).sum()
            false_negatives += ((preds != y) & (y == 1)).sum()
            num_pixels += torch.numel(preds)

    #avg_loss = total_loss / len(loader)
    avg_loss = total_loss / len(loader.dataset)
    if device == 0:
        print(
            f"\n\nGot {num_correct}/{num_pixels} correct pixels, with TP = {true_positives}, TN = {true_negatives}, FP = {false_positives}, FN = {false_negatives}."
            )
        curr_miou = ((true_positives/(true_positives + false_positives + false_negatives)) + (true_negatives/(true_negatives + false_negatives + false_positives)))/2
        curr_f1_score = (2 * true_positives)/(2 * true_positives + false_positives + false_negatives)

        if curr_miou > best_miou:
            best_miou = curr_miou.item()
            save_best_miou(checkpoint)
        
        if curr_f1_score > best_f1_score:
            best_f1_score = curr_f1_score.item()
            save_best_f1_score(checkpoint)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_best_loss(checkpoint)
        
        avg_loss_list.append(avg_loss)
        save_json_score()

        #if device == 0:
        print(
            f"""\nMetric scores: Accuracy = {(true_positives + true_negatives)/num_pixels}\n 
            Precision = {true_positives/(true_positives + false_positives)}\n 
            Recall = {true_positives/(true_positives + false_negatives)}\n 
            F1-Score = {curr_f1_score}\n 
            mIoU = {curr_miou}"""
        )
    model.train() # Set the module in training mode
    return avg_loss

def save_predictions_as_imgs(
    loader, model, gpu_id, folder="saved_images/", device="cuda"
):
    if not os.path.exists(folder):
        print("os.path.exists(folder) deu falso")
        os.makedirs(folder)

    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            #preds = torch.sigmoid(model(x))
            preds = model(x)
            preds_5 = (preds > 0.5).float()
            preds_6 = (preds > 0.6).float()
            preds_7 = (preds > 0.7).float()
            preds_8 = (preds > 0.8).float()
            
        torchvision.utils.save_image(
            preds_5, f"{folder}pred_5_gpu{gpu_id}_{idx}.png"
        )
        torchvision.utils.save_image(
            preds_6, f"{folder}pred_6_gpu{gpu_id}_{idx}.png"
        )
        torchvision.utils.save_image(
            preds_7, f"{folder}pred_7_gpu{gpu_id}_{idx}.png"
        )
        torchvision.utils.save_image(
            preds_8, f"{folder}pred_8_gpu{gpu_id}_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}_gpu{gpu_id}_{idx}.png")

    model.train()