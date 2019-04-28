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

# Global variables
shape_path = 'trained_model/model_shape.json'
weights_path = 'trained_model/model_weights.h5'
img_dims = 512
#/home/niehusst/vision262/project/YOLO/data/TCGA-61-2012/1.3.6.1.4.1.14519.5.2.1.6450.4007.336565074650874040486975138397/18

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

    # ensure image is grayscale (only has 1 channel)
    im_adjusted = im_adjusted.astype(np.float32)
    if len(im_adjusted.shape) >= 3:
        # squash 3 channel image
        im_adjusted = np.dot(im_adjusted[...,:3], [0.299, 0.587, 0.114])
    
    # normalize the image to a 0-1 range
    if not np.amax(im_adjusted) < 1: # check that image isn't already normalized
        if np.amin(im_adjusted) < 0:
            im_adjusted += np.amin(im_adjusted)
        im_adjusted /= np.amax(im_adjusted)
    
    # model requires 4D input; shape it to expected dims
    im_adjusted = np.reshape(im_adjusted, (1, img_dims, img_dims, 1))
    return im_adjusted

def main(argv):
    """
    Loads a saved Keras model from the trained_model/ directory and loads the
    image from the path specified in the command line argument. It uses the 
    model to make a bounding box prediction on the input image, and displays the
    image with the predicted bounding box.
    @param agrv[1] - command line argument, path to valid CT scan img containing
                     cancer which the model can make a prediction on.
    """
    # load a pretrained model from HDF5 file
    model = load_model(shape_path, weights_path)

    # load image from argv
    img = load_image(argv[1])
    
    # ensure image fits model input dimensions
    preprocessed_img = pre_process(img)
    
    # make a prediction on the loaded image
    output = model.predict(preprocessed_img, batch_size=1)
    
    # un-normalize prediction to get plotable points
    points = np.array(output[0]) * 512
    points = list(points.astype(np.int32))
    print('Predicted points: {}'.format(points))

    # display prediction on image (with ground truth if training data???)
    #TODO:how to draw the bb?
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    draw.rectangle(points, outline='#ff0000')
    im.show() #TODO: loading this way causes the display range to be bad, normalized is too squeezed a range (compare to command line 'display')

"""
A program to use the trained and saved YOLO cancer detection model to make a 
bounding box prediction on a single image.
"""
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: " + sys.argv[0] + " <image file path>")
    else:
        main(sys.argv)
