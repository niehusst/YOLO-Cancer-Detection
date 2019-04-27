# Import tensorflow
import tensorflow as tf
import keras.backend as K

# Helper libraries
import math
import numpy as np
import pandas as pd
import pydicom
import os
import sys
import random

# Imports for dataset manipulation
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# Improve progress bar display
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

tf.enable_eager_execution() #comment this out if causing errors
tf.logging.set_verbosity(tf.logging.DEBUG)

"""
NOTES:
use tf.image.nonMaxSuppression (perform non max sup on of bounding boxes of intersection over the union)

use tf.image.draw_bounding_boxes (draws bb points on images in passed in tensor objects of pts and imgs)

"""

###         GET THE DATASET AND PREPROCESS IT        ###

# relative paths to image data and labels
CSV_PATH = 'CCC_clean.csv'
IMAGE_BASE_PATH = '../data/'

print("Loading and processing data\n")

data_frame = pd.read_csv(CSV_PATH)

# zip points data together
points = zip(data_frame['start_x'], data_frame['start_y'], \
                       data_frame['end_x'], data_frame['end_y'])
img_paths = data_frame['imgPath']
# try to classify the area of the body the tumor was in too???? using 'anatomy' col 


# do some preprocessing of the data
def path_to_image(path):
    #load image from path as numpy array
    image = pydicom.dcmread(os.path.join(IMAGE_BASE_PATH, path)).pixel_array
    return image

# normalize dicom image pixel values to 0-1 range
def normalize_image(img):
    img = img.astype(np.float32)
    img += abs(np.amin(img))
    img /= np.amax(img)
    return img

# normalize the ground truth bounding box labels wrt image dimensions
def normalize_points(points):
    imDims = 512.0 #each image is 512x512 
    points = list(points)
    for i in range(len(points)):
        points[i] /= imDims
    return np.array(points).astype(np.float32)

# apply preprocessing functions
points = map(normalize_points, points)
imgs = map(path_to_image, img_paths) 
imgs = map(normalize_image, imgs)

# reshape input image data to expected 4D shape and cast all data to np arrays
imgs = np.array(imgs)
points = np.array(points)
imgs = imgs.reshape(-1, 512, 512, 1)

# split the preprocessed data into train and test
train_imgs, test_imgs, train_points, test_points = \
  train_test_split(imgs, points, test_size=0.15, random_state=42)

num_train_examples = len(train_imgs)
num_test_examples = len(test_imgs)


# create generator for training the model in batches
generator = ImageDataGenerator(rotation_range=0, zoom_range=0,
	width_shift_range=0, height_shift_range=0, shear_range=0,
	horizontal_flip=False, fill_mode="nearest")
#TODO: use data augment to flip? (change steps per batch so all images get seen in an epoch!)


print("Data preprocessing complete\n")


###            DEFINITION OF MODEL SHAPE             ###

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (7, 7), padding='same', activation=tf.nn.relu,
                               strides=2, input_shape=(512, 512, 1)),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),

    tf.keras.layers.Conv2D(192, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),

    tf.keras.layers.Conv2D(128, (1,1), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(256, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(256, (1,1), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(512, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),

    tf.keras.layers.Conv2D(256, (1,1), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(512, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(256, (1,1), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(512, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(256, (1,1), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(512, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(256, (1,1), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(512, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(512, (1,1), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),

    tf.keras.layers.Conv2D(512, (1,1), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(512, (1,1), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(1024, (3,3), padding='same', strides=2, activation=tf.nn.relu),

    tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.nn.relu),

    tf.keras.layers.Flatten(), #flatten images into array for the fully connnected layers
#    tf.keras.layers.Dense(1024, activation=tf.nn.sigmoid),
    #tf.keras.layers.Dropout(0.5), # prevents overfitting for large number of epochs?
#    tf.keras.layers.Dense(4096, activation=tf.keras.activations.linear),
    tf.keras.layers.Dense(4, activation=tf.nn.sigmoid) # 4 outputs: predict 4 points for a bounding box
])
"""
Our final layer predicts both class probabilities and
bounding box coordinates. We normalize the bounding box
width and height by the image width and height so that they
fall between 0 and 1.

We use a sigmoid activation function for the final layer to facilitate
learning of the  normalized range of the output.
all other layers use the following leaky rectified linear activation:
x if x>0 else 0.1*x
(i think tf.nn.leaky_relu has default of 0.2 instead of 0.1)
"""

# custom loss function using aspects of relevant information from the YOLO paper
def YOLO_loss(y_true, y_pred):
    # extract points from tensors:
    # ground truth label points
    x_LT = y_true[:, 0] # left x coord
    y_UT = y_true[:, 1] # upper y coord
    x_RT = y_true[:, 2] # right x coord
    y_LT = y_true[:, 3] # lower y coord
    # predicted points
    x_LP = y_pred[:, 0] # left x coord
    y_UP = y_pred[:, 1] # upper y coord
    x_RP = y_pred[:, 2] # right x coord
    y_LP = y_pred[:, 3] # lower y coord 

    lambda_coord = 5

    # calculate the mean squared error for mid_points
    x_Pmid = tf.math.add(x_LP, tf.math.divide(tf.math.subtract(x_RP, x_LP), 2))
    x_Tmid = tf.math.add(x_LT, tf.math.divide(tf.math.subtract(x_RT, x_LT), 2))
    y_Pmid = tf.math.add(y_UP, tf.math.divide(tf.math.subtract(y_LP, y_UP), 2))
    y_Tmid = tf.math.add(y_UT, tf.math.divide(tf.math.subtract(y_LT, y_UT), 2))

    x_mid_sqdiff = tf.math.square(tf.math.subtract(x_Pmid, x_Tmid))
    y_mid_sqdiff = tf.math.square(tf.math.subtract(y_Pmid, y_Tmid))

    first_term = tf.math.add(x_mid_sqdiff, y_mid_sqdiff)

    # calculate mean squared error for width and height
    x_Pwidth = tf.math.sqrt(tf.math.abs(tf.math.subtract(x_RP, x_LP)))
    x_Twidth = tf.math.sqrt(tf.math.abs(tf.math.subtract(x_RT, x_LT)))
    y_Pheight = tf.math.sqrt(tf.math.abs(tf.math.subtract(y_UP, y_LP)))
    y_Theight = tf.math.sqrt(tf.math.abs(tf.math.subtract(y_UT, y_LT)))

    second_term = tf.math.add(tf.math.square(tf.math.subtract(x_Pwidth,  x_Twidth)),
                              tf.math.square(tf.math.subtract(y_Pheight, y_Theight)))

    # calculate the intersection over the union
    intersection = tf.math.multiply(tf.math.abs(tf.math.subtract(x_LT, x_RP)),
                                        tf.math.abs(tf.math.subtract(y_UT, y_LP)))
    union_double = tf.math.add(tf.math.multiply(tf.math.subtract(x_RP, x_LP), tf.math.subtract(y_LP,y_UP)),
                               tf.math.multiply(tf.math.subtract(x_RT, x_LT), tf.math.subtract(y_LT, y_UT)))
    union = tf.math.abs(tf.math.subtract(union_double, intersection))
    iou = tf.math.divide(intersection, union) #TODO: IOU actually increases as match is better; it should decrease+
    #invert and add epsilon value (some super tiny thing)

    loss = tf.math.add(tf.math.multiply(tf.math.add(first_term, second_term), lambda_coord), iou)

    return iou


#TODO: adjust parameters for adam optimizer; change learning rate?
model.compile(optimizer='adam',
              loss='mean_squared_error',#YOLO_loss,
              metrics=['accuracy'])

#print(model.summary()) #see the shape of the model


###                   TRAIN THE MODEL                ###
"""
We train the network for about 135 epochs(thats a lot, they required dropout and data aug).
Throughout training we use a batch size of 64, a momentum of 0.9 and a decay of 0.0005.

Our  learning  rate  schedule  is  as  follows:  For  the  first epochs
we slowly raise the learning rate from 10e-3 to 10e-2. If we start at a
high learning rate our model often diverges due to unstable gradients.
We continue training with 10e-2 for 75 epochs, then 10e-3 for 30 epochs,
and finally 10e-4 for 30 epochs
"""
BATCH_SIZE = 5
num_epochs = 1

print('fitting the model\n')
model.fit_generator(generator.flow(train_imgs, train_points, batch_size=BATCH_SIZE), epochs=num_epochs, \
    steps_per_epoch=(num_train_examples // BATCH_SIZE))



###                 EVALUATE THE MODEL               ###

#evaluate the accuracy of the trained model using the test datasets
loss, accuracy = model.evaluate(test_imgs, test_points)
print("Final loss:{}\nFinal accuracy:{}".format(loss, accuracy))



###                 SAVING THE MODEL                 ###
# save the model so that it can be loaded without training later
shape_path = 'trained_model/model_shape.json'
weight_path = 'trained_model/model_weights.h5'
#model.save(save_path, overwrite=True) #broken#save entire model as HDF5 model

# serialize model to JSON
model_json = model.to_json()
with open(shape_path, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(weight_path)
print("Saved model to disk")

