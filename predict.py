# Import tensorflow 
import tensorflow as tf
import keras

# Imports for visualizing predictions
import numpy as np
import pydicom
from skimage.transform import resize
import PIL
from PIL import Image, ImageDraw, ImageColor

# Helper imports
import sys, os
import pandas as pd
import time

# Path variables 
shape_path = 'trained_model/model_shape.json'
weights_path = 'trained_model/model_weights.h5'
CSV_PATH = 'label_data/CCC_clean.csv'
IMAGE_BASE_PATH = '../data/'

# Global variables
img_dims = 512


def load_model(shape_file, weights_file):
    """
    Load a tensorflow/keras model from an HDF5 file found at provided path.

    @param path - path to valid HDF5 file of YOLO cancer detection model
    @return model - a fully trained tf/keras model
    """
    print("Loading model from disk..."),
    # load json and create model
    json_file = open(shape_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_file)
    print("Complete!")
    return loaded_model

def is_dicom(im_path):
    """
    Check if the image specified by the path is a DICOM image.

    @param im_path - string path to an image file
    @return - boolean, True if is a dicom image, else False
    """
    # get file extension
    path, ext = os.path.splitext(im_path)
    
    # if file extension is empty or is .dcm, assume DICOM file
    # (we assume empty extension means DICOM because the format
    # of the DICOM data we are using was saved with an empty file
    # extension)
    if ext == ".dcm" or ext == "":
        return True
    else:
        return False

def load_image(im_path):
    """
    Load an image from provided path. Loads both DICOM and more
    common image formats into a numpy array.

    @param im_path - string path to the image file
    @return im - the image loaded as a numpy ndarray
    """
    if is_dicom(im_path):
        # load with pydicom
        im = pydicom.dcmread(im_path).pixel_array
        return im
    else:
        # load with Pillow
        im = Image.open(im_path)
        return np.array(im)

def pre_process(img):
    """
    Takes an image and preprocesses it to fit in the model

    @param img - a numpy ndarray representing an image
    @return - a shaped and normalized, grayscale version of img
    """
    # resize image to 512x512
    im_adjusted = resize(img, (img_dims,img_dims), anti_aliasing=True,\
                             preserve_range=True)

    # ensure image is grayscale (only has 1 channel)
    im_adjusted = im_adjusted.astype(np.float32)
    if len(im_adjusted.shape) >= 3:
        # squash 3 channel image to grayscale
        im_adjusted = np.dot(im_adjusted[...,:3], [0.299, 0.587, 0.114])
    
    # normalize the image to a 0-1 range
    if not np.amax(im_adjusted) < 1: # check that image isn't already normalized
        if np.amin(im_adjusted) < 0:
            im_adjusted += np.amin(im_adjusted)
        im_adjusted /= np.amax(im_adjusted)
    
    # model requires 4D input; shape it to expected dims
    im_adjusted = np.reshape(im_adjusted, (1, img_dims, img_dims, 1))
    return im_adjusted

def normalize_image(img):
    """
    Normalize an image to the range of 0-255. This may help reduce the white
    washing that occurs with displaying DICOM images with PIL.

    @param img - a numpy array representing an image
    @return normalized - a numpy array whose elements are all within the range 
                         of 0-255
    """
    # adjust for negative values
    normalized = img + np.abs(np.amin(img))
    
    # normalize to 0-1
    normalized = normalized.astype(np.float32)
    normalized /= np.amax(normalized)
    
    # stretch scale of range to 255
    normalized *= 255
    return normalized
    
def main():
    """
    Loads a saved Keras model from the trained_model/ directory and loads the
    image and ground truth points from the loaded dataset. It uses the 
    model to make a bounding box prediction on the input image, and displays the
    image with the predicted and true bounding boxs.
    """
    # load a pretrained model from HDF5 file
    model = load_model(shape_path, weights_path)

    # load dataset for iterating paths
    data_frame = pd.read_csv(CSV_PATH)

    # iterate over all image paths
    for i in range(len(data_frame['imgPath'])):
        img = load_image(IMAGE_BASE_PATH + data_frame['imgPath'][i])
    
        # ensure image fits model input dimensions
        preprocessed_img = pre_process(img)
    
        output = model.predict(preprocessed_img, batch_size=1)
        
        # un-normalize prediction to get plotable points
        points = np.array(output[0]) * 512
        points = list(points.astype(np.int32))
        
        # normalize image to prevent as much white-washing caused
        # by PIL lib as possible
        norm = normalize_image(img)
        
        # draw bbox of predicted points
        im = Image.fromarray(norm).convert("RGB")#convert RGB for colored bboxes
        draw = ImageDraw.Draw(im)
        draw.rectangle(points, outline='#ff0000') #red bbox
        
        #draw bbox of ground truth
        true_points = [int(data_frame['start_x'][i]),
                       int(data_frame['start_y'][i]),
                       int(data_frame['end_x'][i]),
                       int(data_frame['end_y'][i])]
        draw.rectangle(true_points, outline='#00ff00') #green bbox
        
        im.show()
        time.sleep(1) #sleep to let user see/close image
        

"""
A program to use the trained and saved YOLO cancer detection model to make a 
bounding box prediction one image at time from the dataset specified by the 
data path global, iterating and showing ground truth bbox as well
"""
if __name__ == '__main__':
    main()
