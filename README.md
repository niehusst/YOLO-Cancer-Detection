# YOLO-Cancer-Detection

An implementation of the YOLO algorithm trained to spot tumors in DICOM images. The model is trained on the "Crowds Cure Cancer" dataset, which only contains images that DO have tumors; this model will always predict a bounding box for a tumor (even if one is not present).

### Getting Started
We recommend using a virtual environment to ensure your dependencies are up-to-date and freshly installed. From there, you can use `pip` to install all the dependencies in the `deps.txt` file (this can be done quickly with `pip install -r deps.txt`).

### Downloading the data
Run the following command from your terminal to download the data.
`$ wget https://www.kaggle.com/kmader/crowds-cure-cancer-2017/downloads/crowds-cure-cancer-2017.zip/3`

### Training the model
`$ python model.py`

### Making predictions
`$ python predict.py`

## Authors
* **Liam Niehus-Staab** - [niehusst](https://github.com/niehusst)
* **Eli Salm** - [salmeli](https://github.com/salmeli)

## Aknowledgements
* The "Crowds Cure Cancer" dataset used to train the model in this repo can be found on Kaggle [here](https://www.kaggle.com/kmader/crowds-cure-cancer-2017)
* The YOLO algorithm used in this project was developed by Redmond et. al. is described in paper found [here](https://arxiv.org/pdf/1506.02640.pdf) 
