import torch
import json
import os
import torchvision
from dataset import ThreeClassCrackTree260
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

best_f1_score = [0,0,0]
best_miou = [0,0,0]
best_loss = 4.0
avg_loss_list = []

def create_json_scores_file():
    data = { "best_f1_score" : [0.0, 0.0, 0.0],
        "best_miou" : [0.0, 0.0, 0.0], 
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

'''def save_best_f1_score(state, filename="best_f1_score.pth.tar"):
    print("=> Saving best f1_score checkpoint.")
    torch.save(state, filename)
    #save_json_score()'''

'''def save_best_miou(state, filename="best_miou.pth.tar"):
    print("=> Saving best miou checkpoint.")
    torch.save(state, filename)    
    #save_json_score()'''

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
    train_ds = ThreeClassCrackTree260(
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

    val_ds = ThreeClassCrackTree260(
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

def check_accuracy(loader, model, checkpoint, loss_fn, device, epsilon=1e-7):
    global best_f1_score, best_miou, best_loss, avg_loss_list

    num_correct = 0
    num_pixels = 0
    true_positives = [0, 0, 0]
    true_negatives = [0, 0, 0]
    false_positives = [0, 0, 0]
    false_negatives = [0, 0, 0]
    total_loss = 0
    model.eval() # Set the module in evaluation mode

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            #y = y.to(device).unsqueeze(1)
            y = y.to(device).long()
            #preds = torch.sigmoid(model(x))
            preds = model(x)
            #probs = torch.sigmoid(preds)
            probs = torch.argmax(preds, dim=1)
            #probs = (probs > 0.5).float()

            #total_loss += loss_fn(probs, y).item()  
            total_loss += loss_fn(preds, y).item() * x.size(0)

            num_correct += (probs == y).sum()
            num_pixels += torch.numel(probs)

            for cls in range(3):
                preds_cls = (probs == cls)
                targets_cls = (y == cls)
                
                true_positives[cls] += (preds_cls & targets_cls).sum().item()
                false_positives[cls] += (preds_cls & (~targets_cls)).sum().item()
                false_negatives[cls] += ((~preds_cls) & targets_cls).sum().item()
                true_negatives[cls] += ((~preds_cls) & (~targets_cls)).sum().item()

    #avg_loss = total_loss / len(loader)
    avg_loss = total_loss / len(loader.dataset)
    if device == 0:

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_best_loss(checkpoint)

        avg_loss_list.append(avg_loss)

        print(f"\n\nGot {num_correct}/{num_pixels} correct pixels\n")
        for cls in range(3):
            if cls == 0:
                print("BACKGROUND PREDICTIONS:")
            elif cls == 1:
                print("CRACK PREDICTIONS:")
            else:
                print("CLOSE TO CRACK PREDICTIONS:")

            print(f"TP = {true_positives[cls]}, TN = {true_negatives[cls]}, FP = {false_positives[cls]}, FN = {false_negatives[cls]}."
            )
            curr_miou = ((true_positives[cls]/(true_positives[cls] + false_positives[cls] + false_negatives[cls])) + (true_negatives[cls]/(true_negatives[cls] + false_negatives[cls] + false_positives[cls])))/2
            curr_f1_score = (2 * true_positives[cls])/(2 * true_positives[cls] + false_positives[cls] + false_negatives[cls])

            if curr_miou > best_miou[cls]:
                best_miou[cls] = curr_miou
                #save_best_miou(checkpoint)
            
            if curr_f1_score > best_f1_score[cls]:
                best_f1_score[cls] = curr_f1_score
                #save_best_f1_score(checkpoint)

            #if device == 0:
            print(
                f"""\nMetric scores: Accuracy = {(true_positives[cls] + true_negatives[cls])/num_pixels}\n 
                Precision = {true_positives[cls]/(true_positives[cls] + false_positives[cls] + epsilon)}\n 
                Recall = {true_positives[cls]/(true_positives[cls] + false_negatives[cls] + epsilon)}\n 
                F1-Score = {curr_f1_score}\n 
                mIoU = {curr_miou}"""
            )
                
        save_json_score()

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