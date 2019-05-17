# YOLO-Cancer-Detection

An implementation of the YOLO algorithm trained to spot tumors in DICOM images. The model is trained on the "Crowds Cure Cancer" dataset, which only contains images that DO have tumors; this model will always predict a bounding box for a tumor (even if one is not present).
**NOTE:** Our final model didn't quite work; we hypothesize that the quality of the data may be one of the culprits in our model's inability to learn how to detect the tumors with any accuracy.

### Getting Started
We recommend using a virtual environment to ensure your dependencies are up-to-date and freshly installed. From there, you can use `pip` to install all the dependencies in the `deps.txt` file (this can be done quickly with `pip install -r deps.txt`).

### Directory Structure

We have included the appropriate directory structure below. This will allow model.py to access the data necessary for training the model. In order to run the following commands, please set up the directory like so:

```
| YOLO/
|---| crowds-cure-cancer-2017/ ## <-- result of Kaggle download
|---|---| annotated_dicoms.zip
|---|---| compressed_stacks.zip
|---|---| CrowdsCureCancer2017Annotations.csv ## <---- @move this file to YOLO/YOLO-Cancer-Detection/label_data
|---| data/
|---|---| TCGA-09-0364
|---|---| ...
## put the data files into this 'data/' directory
|---|---| ...
|---|---| TCGA-OY-A56Q
|---| YOLO-Cancer-Detection/
|---|---| label_data/
|---|---|---| clean_data.py
|---|---|---| CCC_clean.py
|---|---|---| CrowdsCureCancer2017Annotations.csv ## <---- @Here
|---|---| model.py
|---|---| predict.py
|---|---| deps.txt
|---|---| README.md
|---|---| trained_model/ ## <-- THIS IS A DIRECTORY FOR SAVING YOUR MODEL. DO NOT FORGET THIS.
```

### Downloading and cleaning the data
The data must be downloaded directly from [Kaggle](https://www.kaggle.com/kmader/crowds-cure-cancer-2017), where you need to create a username and password (if you don't already have one) in order to download the dataset. Once you have downloaded and unzipped the dataset, you will have the raw images and CSV data. We clean the CSV data down to only the necessary information using the `clean_data.py` script in the `label_data/` directory, which produces a new, clean CSV file which is used in the training and example usage usage of the model.

### Training the model
To train the model, one can simply run `$ python model.py` at the command line. With the adjustable parameters at the top of the file, you can change some aspects of how the model trains very easily. As the model trains, (assuming you have the Tensorboard variables enabled in `model.py`) it will periodically save logged events to the `logs` directory which can be viewed using Tensorboard. Once the model is trained it will be saved to the `trained_model/` directory (note that this directory must exist prior to the model being saved there, as the write command will fail if the directory is not present). 

### Running a quick test
To see the results of your saved model, simply run `$ python predict.py`. This is a simple script that loads up the image data, CSV data, and a trained model from the `trained_model/` directory and allows you to visually compare the predicted and ground truth bounding boxes on each image in the dataset.

## Authors
* **Liam Niehus-Staab** - [niehusst](https://github.com/niehusst)
* **Eli Salm** - [salmeli](https://github.com/salmeli)

## Aknowledgements
* The "Crowds Cure Cancer" dataset used to train the model in this repo can be found on Kaggle [here](https://www.kaggle.com/kmader/crowds-cure-cancer-2017)
* The YOLO algorithm used in this project was developed by Redmond et. al. is described in paper found [here](https://arxiv.org/pdf/1506.02640.pdf) 
