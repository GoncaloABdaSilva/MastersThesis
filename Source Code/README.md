# User Guide
This directory contains the source code for all architectures explored in this work.

Training (including validation) is performed using the Kaggle platform ([kaggle.com](https://www.kaggle.com/)), while testing is done locally. 
A Kaggle account is required.

![image alt](https://github.com/GoncaloABdaSilva/MastersThesis/blob/e3545e2bc7e326d723fe61819225da227bd1cd6e/readmeImages/kaggle_welcome.png)

## Dataset Preparation and Upload
The training and validation datasets must be prepared before uploading them to Kaggle.
Here are the links from which some of the pavement crack datasets used in my thesis can be downloaded:
- Liu et al. DeepCrack: [GitHub](https://github.com/yhlleo/DeepCrack);
- Zou et al. DeepCrack (includes CrackTree260, CRKWH100, CrackLS315,  Stone331): [GitHub](https://github.com/qinnzou/DeepCrack);
- CrackForest (CFD): [Dataset](https://github.com/cuilimeng/CrackForest-dataset);
- CrackSC: [GitHub](https://github.com/jonguo111/Transformer-Crack);
- Crack500, GAPs384 and several others: [GitHub](https://github.com/fyangneil/pavement-crack-detection).

### CrackTree260 Preparation
After downloading and unzipping both "image" and "ground truth" folders (from link above), go to "image" and delete the "image.rar" folder.
Because some image files have the extension ".JPG", we convert them to ".jpg". Download this file ([convert_JPG_to_jpg.py](https://github.com/GoncaloABdaSilva/MastersThesis/blob/8eeef90637db71ddbf02162d4c8cfeb46a9c4074/Dataset%20Preparation/CrackTree260/convert_JPG_to_jpg.py)) and insert it into the "image" folder. Open the file in your prefered IDE (VSCode, for example) and run it.

Ground truth images come in ".bmp" format. We convert them into ".png" in order to reduce its memory size, from 130MB into 903KB. In the same directory as "gt", create a new folder named "new_gt". Then, download this file ([convert_bmp_to_png.py](https://github.com/GoncaloABdaSilva/MastersThesis/blob/8eeef90637db71ddbf02162d4c8cfeb46a9c4074/Dataset%20Preparation/CrackTree260/convert_bmp_to_png.py)) and insert it into the same directory "gt" and "new_gt". Open the file in your prefered IDE (VSCode, for example) and run it. After it's done, delete the "gt" folder and rename the folder "new_gt" to "gt".

From the 260 images (and respective ground truths), we have divided them into three folders: 200 images for the "train" folder, 20 images for the "val" folder and 40 images for the "test" folder.
Each of them should have two inside folders: "img" and "gt", where images and ground truth images, respectively, should be. Corresponding image and ground truths will have the same name with a different extension (".jpg" and ".png"). 

#### CrackTree260 with multi-class Preparation
Create a new folder and copy (don't delete) all three previously created folders ("train", "val", "test"). 
Then, download the provided python file ([add_boundary_class.py](https://github.com/GoncaloABdaSilva/MastersThesis/blob/8d6dd7623d2628ff13ca9a001e1d02a4af5c6e0b/Source%20Code/U-Net/Multi-Class%20U-Net/add_boundary_class.py)) and insert it in the same directory as the three folders.
Also in that directory, create three new folders named "new_train", "new_test" and "new_val". 
From the "train" folder, copy its "img" folder into the "new_train" folder. Then create an empty "gt" folder next to it. Do the same for the remaining two folders.
Open the file in your prefered IDE (VSCode, for example) and run it, which will add a neighbourhood class (in red) to the ground truth and insert it into the corresponding "new_gt" folder.
Delete "train", "val" and "test" folders. Rename the "new_X" folders by removing the "new_" from their name.
You should end up with three folders ("train", "val", "test"), each with an "img" and "gt" folder, and inside the "gt" ones, ground truth images in black, white and red.

### Crack500 Preparation
After downloading (from link above), unzipping, and moving the CRACK500 folder to a different directory, delete all three ".txt" files as well as ZIP folders "testdata", "valdata" and "fphb_testresult". Unzip the remaining three folders and delete the ZIP files.
Then inside each of the three folders, create two empty folders named "img" and "gt". Move all images (should be ".jpg" files) into "img" and all ground truths (should be ".png" files) into "gt".
Rename the folders from "testcrop", "traincrop" and "valcrop" by removing "crop" from their name.

### GAPs384 Preparation


### Dataset of images with shadows and CLAHE Preparation

### Upload dataset to Kaggle
A folder should be created with the name of the dataset, having two inside folders: "train" and "val" (previously explained how to be created). 
Once the entire folder structure is set, it must be compressed into a ZIP file. 
On Kaggle, go to "Datasets", click "+ New Dataset", drag and drop the ZIP file and assign it a name.

## Model Code Upload
For the actual model, four python files and one json file are needed: 
- main.py;
- dataset.py;
- utils.py;
- model.py;
- [scores.json](https://github.com/GoncaloABdaSilva/MastersThesis/blob/adedb57853bb7b9889e3b4417b62f5ceaa579b59/Source%20Code/scores.json).

On Kaggle, go to "Datasets", press "+ New Dataset", drag and drop the files and name your dataset accordingly. 
All python files are edited locally (not in Kaggle), so they should first be downloaded into a local directory (along with testing file "test_models"). In case of code modifications, updated files can be re-uploaded by using the "three dots button" on the top right corner, choosing "New Version", and drag and drop the updated files.

In PoolingCrack, the backbone is pre-trained (check [here](https://github.com/GoncaloABdaSilva/MastersThesis/tree/e295d6396b158f3d9eb5520b5bedf69d1d3cd0dc/Source%20Code/PoolingCrack/Pre-train%20Backbone)). A new notebook should be created, and the code from the provided notebook should be copied into it. A separate dataset containing the remaining Python files from the directory must also be created. After pre-training, download the resulting .pth.tar file and upload it to the PoolingCrack code dataset. Further details on how the download and upload procedure are provided later.

In SAM2, instead of "model.py", it's "sam2_seg_wrapper.py", and the notebook code is different (check [sam2-fine-tuning.ipynb](https://github.com/GoncaloABdaSilva/MastersThesis/blob/463055eee4503034dbc181c05a1622ea4adae3c1/Source%20Code/SAM2/sam2-fine-tuning.ipynb)). 

In YOLO, there is only one file, [yolo12-seg-fine-tune.ipynb](https://github.com/GoncaloABdaSilva/MastersThesis/blob/463055eee4503034dbc181c05a1622ea4adae3c1/Source%20Code/YOLOv11-seg/yolo12-seg-fine-tune.ipynb), that should be copied instead. 

## Notebook Creation and Configuration
After uploading both the code and datasets, go to Kaggle's homepage, select "Code", and click "+ New Notebook" to create a notebook. All training is executed within this notebook, while using Kaggle's free computational resources. 
To use the uploaded files, click "+ Add input", filter by "Your Work" and "Datasets" to reveal all your datasets, and press the "+" button on those to be used.

For dual-GPU training (16GB x2), under the notebook's name go to "Settings", "Accelerator", "GPU T4 x2". These resources are limited to 30 hours per week. You can check your weekly quota by going to Kaggle's homepage and clicking on the icon on the top right corner.
The file [notebook_example.ipynb](https://github.com/GoncaloABdaSilva/MastersThesis/blob/3d9007e7450842499506f1f5639a5c61f46a4643/Source%20Code/notebook_example.ipynb) contains the code used to start training , as well as an example of what it prints while running. Copy the code cells to your notebook.

The following image is an example of how it should look.
![image alt](https://github.com/GoncaloABdaSilva/MastersThesis/blob/e3545e2bc7e326d723fe61819225da227bd1cd6e/readmeImages/kaggle_notebook.png)

## Dataset Path Configuration
Your dataset names might have different names from mine, making their paths in kaggle different. You can look up for them by hovering their name and pressing "Copy file path". 
If image dataset paths are different, then data directories must be updated in train.py by copying the new file path into a variable such as "CRACKTREE260_DS_KAGGLE", for example. Update the dataset path at the first cell (you may also delete it, as it is not needed).
If code dataset paths are different, update path at notebook's third cell, both in second and last line.

On the third cell you must state for how many epochs you would like to train the model by changing the number after "--epochs".

## Training Execution Options
To train the model there are two options:
- The first one is by pressing the button "Run all", which will automatically start a session and execute all cells in order. Each cell's prints will appear between that cell and the next one, and output files will appear on the Output section. 
- The second option is by pressing the "Save Version" button on the top right corner. After giving a name (e.g. "Attempt 1), select "Save & Run All" and on advanced settings select "Run with GPU for all sessions".

The main disadvantages from the first option to the second are:
- The notebook requires to be actively used during training while the second doesn't. If the session is stopped all output files will be lost rather than being permanently accessible after the session is complete;
- If code is updated, it requires to manually scan for updates by hovering the dataset name, "More Options", "Check for updates".
The second option has the inconvenience that once a Version is created, although it can be stopped and the name can be changed, it cannot be deleted.

## Training Output Files and Re-training
During training, we validate learning using metrics like F1-score and mIoU. In epochs where we get the best metric score, we save a copy of the model at that epoch. In the end we end up with four model files, with the extension ".pth.tar" (e.g. "best_miou.pth.tar"). A JSON file is also created (scores.json), containing metric values for every epoch as well as top metric scores.
Once training is over, files can be downloaded. 
- In the "Run All" option, they are available in Output, on the right side of the screen.
- In the "Save Version" option, you must go to Kaggle's homepage, select "Code", access your notebook and click on "Version n of n" ("n" being a number). You will get a list of all versions of that notebook, in cronological order.

If you want to continue with training one of the models, you must upload the desired .pth.tar and updated scores.json files to the dataset with the model's code. Then, on the last line of the notebook, replace "--no-load_model" with "--load-model". The number of epochs must be updated to reflect the total desired number of epochs (e.g. after training for 10 epochs, and training an aditional 20, epochs should be set to 30).

## Testing
For testing, in the same local directory as your code:
- Create a folder named "result" and store the .pth.tar files. You may also store the scores.json file and use it to plot metric scores progression.
- Create a folder (or copy) named "test" with your testing images and respective ground truths, following the same "img" and "gt" structure used previously.
- Create a folder named "test_saved_images". Images produced during validation will be stored here, with one subfolder per model.
- Run "test_models.py" on your prefered IDE, results will be printed.

## Fine-tuning YOLO
For fine-tuning YOLOv11-seg, the user needs to upload the desired image dataset the same way as before, changing its name on the last code cell, to enable the "GPU T4 x2" accelerator, and start training with one of the presented options. It is also possible to change the number of epochs or YOLO model used by changing the variables in code cells.
