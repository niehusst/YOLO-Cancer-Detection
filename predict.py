# Import tensorflow 
import tensorflow as tf
import tensorflow_datasets as tfds
import keras

# Imports for visualizing predictions
import numpy as np
import matplotlib
matplotlib.use('PS') #prevent import error due to venv
import matplotlib.pyplot as plt
from skimage.transform import resize

# Helper imports
import sys, os

# Global variables
shape_path = 'trained_model/model_shape.json'
weights_path = 'trained_model/model_weights.h5'
img_dims = 512


def load_model(path):
    """
    Load a tensorflow/keras model from an HDF5 file found at provided path.
    @param path - path to valid HDF5 file of YOLO cancer detection model
    @return model - a fully trained tf/keras model
    """
    print("Loading model from disk...", end="")
    # load json and create model
    json_file = open(shape_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path)
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
    if ext == ".dcm" or ext == "":
        return True
    else:
        return False

def load_image(im_path):
    """
    Load an image from provided path. Loads both DICOM and more
    common image formats.
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

    #TODO: ensure image is grayscale (only has 1 channel)
    im_adjusted = im_adjusted.astype(np.float32)
    
    #TODO: normalize the image to a 0-1 range
    
    # model requires 4D input; shape it to expected dims
    im_adjusted = np.reshape(im_adjusted, (1, img_dims, img_dims, 1))
    return im_adjusted

def main(argv):
    # load a pretrained model from HDF5 file
    model = load_model(load_path)

    # load image from argv
    img = load_image(argv[1])
    
    # ensure image fits model input dimensions
    img = pre_process(img)
    
    # make a prediction on the loaded image
    output = model.predict(img, batch_size=1)
    print('first prediction was {}'.format(output[0]))

    # display prediction on image (with ground truth if training data???)
    #TODO:use matplotlib to draw the bb? remember predicted pointes are dimension normalized


"""
A program to use the trained and saved YOLO cancer detection model to make a 
bounding box prediction on a single image.
"""
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: " + sys.argv[0] + " <image file path>")
    else:
        main(sys.argv)
