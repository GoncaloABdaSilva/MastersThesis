import torch
import json
import os
import torchvision
from dataset import CrackTree260
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import io
import zipfile

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

def save_json_score():
    #.item() as they are torch.Tensor
    data = { "best_f1_score" : best_f1_score, 
        "best_miou" : best_miou, 
        "best_loss": best_loss,
        "avg_loss_list": avg_loss_list,}
    
    with open("/kaggle/working/scores.json", 'w') as new_jfile:
            json.dump(data, new_jfile)
    
def save_checkpoint(state, filename="/kaggle/working/my_checkpoint.pth.tar"):
    print("\n=> Saving checkpoint")
    torch.save(state, filename)

def save_best_f1_score(state, filename="/kaggle/working/best_f1_score.pth.tar"):
    print("=> Saving best f1_score checkpoint.")
    torch.save(state, filename)

def save_best_miou(state, filename="/kaggle/working/best_miou.pth.tar"):
    print("=> Saving best miou checkpoint.")
    torch.save(state, filename)    

def save_best_loss(state, filename="/kaggle/working/best_loss.pth.tar"):
    print("=> Saving best loss checkpoint.")
    torch.save(state, filename)


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers,
    pin_memory
):
    train_ds = CrackTree260(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_ds = CrackTree260(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader

def get_test_loaders(
    test_dir,
    test_mask_dir,
    batch_size,
    test_transform,
    num_workers,
    pin_memory
):
    test_ds = CrackTree260(
        image_dir=test_dir,
        mask_dir=test_mask_dir,
        transform=test_transform,
    )

    test_ds.images = sorted(test_ds.images)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return test_loader
    

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
            preds = (preds >= 0.5).float()

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

    print(
        f"""\nMetric scores: Accuracy = {(true_positives + true_negatives)/num_pixels}\n 
        Precision = {true_positives/(true_positives + false_positives)}\n 
        Recall = {true_positives/(true_positives + false_negatives)}\n 
        F1-Score = {curr_f1_score}\n 
        mIoU = {curr_miou}\n
        Epoch loss = {avg_loss}"""
    )

    model.train() # Set the module in training mode
    return avg_loss

def test_checkpoint(loader, model, loss_fn, device):
    num_correct = 0
    num_pixels = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    total_loss = 0

    """true_positives_60 = 0
    true_negatives_60 = 0
    false_positives_60 = 0
    false_negatives_60 = 0

    true_positives_70 = 0
    true_negatives_70 = 0
    false_positives_70 = 0
    false_negatives_70 = 0

    true_positives_80 = 0
    true_negatives_80 = 0
    false_positives_80 = 0
    false_negatives_80 = 0

    true_positives_90 = 0
    true_negatives_90 = 0
    false_positives_90 = 0
    false_negatives_90 = 0"""
    model.eval()


    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            #preds = torch.sigmoid(model(x))
            probs = model(x)
            preds = torch.sigmoid(probs)
            
            """preds_60 = (preds >= 0.6).float()
            preds_70 = (preds >= 0.7).float()
            preds_80 = (preds >= 0.8).float()
            preds_90 = (preds >= 0.9).float()"""

            preds = (preds >= 0.5).float()

            #total_loss += loss_fn(probs, y).item()  
            total_loss += loss_fn(probs, y).item() * x.size(0)

            num_correct += (preds == y).sum()
            true_positives += ((preds == y) & (y == 1)).sum()
            true_negatives += ((preds == y) & (y == 0)).sum()
            false_positives += ((preds != y) & (y == 0)).sum()
            false_negatives += ((preds != y) & (y == 1)).sum()
            num_pixels += torch.numel(preds)

            torchvision.utils.save_image(
                preds, f"/kaggle/working/pred_{idx}.png"
            )

            
            """
            true_positives_60 += ((preds_60 == y) & (y == 1)).sum()
            true_negatives_60 += ((preds_60 == y) & (y == 0)).sum()
            false_positives_60 += ((preds_60 != y) & (y == 0)).sum()
            false_negatives_60 += ((preds_60 != y) & (y == 1)).sum()

            torchvision.utils.save_image(
                preds_60, f"/kaggle/working/pred60_{idx}.png"
            )


            true_positives_70 += ((preds_70 == y) & (y == 1)).sum()
            true_negatives_70 += ((preds_70 == y) & (y == 0)).sum()
            false_positives_70 += ((preds_70 != y) & (y == 0)).sum()
            false_negatives_70 += ((preds_70 != y) & (y == 1)).sum()


            torchvision.utils.save_image(
                preds_70, f"/kaggle/working/pred70_{idx}.png"
            )


            true_positives_80 += ((preds_80 == y) & (y == 1)).sum()
            true_negatives_80 += ((preds_80 == y) & (y == 0)).sum()
            false_positives_80 += ((preds_80 != y) & (y == 0)).sum()
            false_negatives_80 += ((preds_80 != y) & (y == 1)).sum()

            torchvision.utils.save_image(
                preds_80, f"/kaggle/working/pred80_{idx}.png"
            )


            true_positives_90 += ((preds_90 == y) & (y == 1)).sum()
            true_negatives_90 += ((preds_90 == y) & (y == 0)).sum()
            false_positives_90 += ((preds_90 != y) & (y == 0)).sum()
            false_negatives_90 += ((preds_90 != y) & (y == 1)).sum()

            torchvision.utils.save_image(
                preds_90, f"/kaggle/working/pred90_{idx}.png"
            )"""

    #avg_loss = total_loss / len(loader)
    avg_loss = total_loss / len(loader.dataset)

    print(
        f"\n\nGot {num_correct}/{num_pixels} correct pixels, with TP = {true_positives}, TN = {true_negatives}, FP = {false_positives}, FN = {false_negatives}."
        )
    curr_miou = ((true_positives/(true_positives + false_positives + false_negatives)) + (true_negatives/(true_negatives + false_negatives + false_positives)))/2
    curr_f1_score = (2 * true_positives)/(2 * true_positives + false_positives + false_negatives)

    print(
        f"""\nMetric scores: Accuracy = {(true_positives + true_negatives)/num_pixels}\n 
        Precision = {true_positives/(true_positives + false_positives)}\n 
        Recall = {true_positives/(true_positives + false_negatives)}\n 
        F1-Score = {curr_f1_score}\n 
        mIoU = {curr_miou}\n
        Test loss = {avg_loss}"""
    )

    #print(f"\n==> Testing with preds >= 0.6")

    #curr_miou = ((true_positives_60/(true_positives_60 + false_positives_60 + false_negatives_60)) + (true_negatives_60/(true_negatives_60 + false_negatives_60 + false_positives_60)))/2
    #curr_f1_score = (2 * true_positives_60)/(2 * true_positives_60 + false_positives_60 + false_negatives_60)

    #print(
     #   f"""\nMetric scores: Accuracy = {(true_positives_60 + true_negatives_60)/num_pixels}\n 
      #  Precision = {true_positives_60/(true_positives_60 + false_positives_60)}\n 
       # Recall = {true_positives_60/(true_positives_60 + false_negatives_60)}\n 
        #F1-Score = {curr_f1_score}\n 
        #mIoU = {curr_miou}\n
        #Test loss = {avg_loss}"""
    #)

    #print(f"\n==> Testing with preds >= 0.7")

    #curr_miou = ((true_positives_70/(true_positives_70 + false_positives_70 + false_negatives_70)) + (true_negatives_70/(true_negatives_70 + false_negatives_70 + false_positives_70)))/2
    #curr_f1_score = (2 * true_positives_70)/(2 * true_positives_70 + false_positives_70 + false_negatives_70)

    #print(
     #   f"""\nMetric scores: Accuracy = {(true_positives_70 + true_negatives_70)/num_pixels}\n 
     #   Precision = {true_positives_70/(true_positives_70 + false_positives_70)}\n 
     #   Recall = {true_positives_70/(true_positives_70 + false_negatives_70)}\n 
     #   F1-Score = {curr_f1_score}\n 
     #   mIoU = {curr_miou}"""
    #)

    #print(f"\n==> Testing with preds >= 0.8")

    #curr_miou = ((true_positives_80/(true_positives_80 + false_positives_80 + false_negatives_80)) + (true_negatives_80/(true_negatives_80 + false_negatives_80 + false_positives_80)))/2
    #curr_f1_score = (2 * true_positives_80)/(2 * true_positives_80 + false_positives_80 + false_negatives_80)

    #print(
     #   f"""\nMetric scores: Accuracy = {(true_positives_80 + true_negatives_80)/num_pixels}\n 
     #   Precision = {true_positives_80/(true_positives_80 + false_positives_80)}\n 
     #   Recall = {true_positives_80/(true_positives_80 + false_negatives_80)}\n 
     #   F1-Score = {curr_f1_score}\n 
     #   mIoU = {curr_miou}"""
    #)

    #print(f"\n==> Testing with preds >= 0.9")

    #curr_miou = ((true_positives_90/(true_positives_90 + false_positives_90 + false_negatives_90)) + (true_negatives_90/(true_negatives_90 + false_negatives_90 + false_positives_90)))/2
    #curr_f1_score = (2 * true_positives_90)/(2 * true_positives_90 + false_positives_90 + false_negatives_90)

    #print(
     #   f"""\nMetric scores: Accuracy = {(true_positives_90 + true_negatives_90)/num_pixels}\n 
     #   Precision = {true_positives_90/(true_positives_90 + false_positives_90)}\n 
     #   Recall = {true_positives_90/(true_positives_90 + false_negatives_90)}\n 
     #   F1-Score = {curr_f1_score}\n 
     #   mIoU = {curr_miou}"""
    #)

