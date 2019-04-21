# Import tensorflow 
import tensorflow as tf
import tensorflow_datasets as tfds
tf.logging.set_verbosity(tf.logging.ERROR)

# Helper libraries
import math
import numpy as np
import pandas as pd
import pydicom as pdcm
import os
import random

# Imports for dataset separation
from sklearn.model_selection import train_test_split

# Improve progress bar display
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

# allow for dataset iteration. 
#tf.enable_eager_execution() #comment this out if causing errors

tf.logging.set_verbosity(tf.logging.DEBUG)
"""
possible causes for squeeze error:
OOM is actually the root cause, but squeeze just arises from it
something to do with collecting accuracy metrics: possibly model is expecting train labels to be scalar?
how do predictions/labels look for other people doing this??
"""
###         GET THE DATASET AND PREPROCESS IT        ###

# relative paths to data and labels
CSV_PATH = 'CCC_clean.csv'
IMAGE_BASE_PATH = '../data/'

print("Loading and processing data\n")

data_frame = pd.read_csv(CSV_PATH)

# zip points data together                        TODO: is this what data should contain???
train_points = zip(data_frame['start_x'], data_frame['start_y'], \
                       data_frame['end_x'], data_frame['end_y'])
train_img_paths = data_frame['imgPath']
# TODO: add class label for all the data (each should just be 1 (aka cancer) since we
# are not doing classification)
# try to classify the area of the body the tumor was in too???? using 'anatomy' col of 

num_train_examples = len(data_frame['imgPath'])

# do some preprocessing of the data
def path_to_image(path):
    #load image from path as numpy array
    image = pdcm.dcmread(os.path.join(IMAGE_BASE_PATH, path)).pixel_array
    return image

# normalize dicom image pixel values to 0-1 range
def normalize_image(img):
    img = img.astype(np.float32)
    img += abs(np.amin(img)) 
    img /= np.amax(img)
    return img

# normalize the ground truth bounding box labels wrt image dimensions
def normalize_points(points):
    imDims = 512.0 # TODO dont hardcode??
    points = list(points)
    for i in range(len(points)):
        points[i] /= imDims
    return np.array(points).astype(np.float32)

train_points = map(normalize_points, train_points)
train_imgs = map(path_to_image, train_img_paths) 
train_imgs = map(normalize_image, train_imgs)
#poop = np.array(range(len(train_points)))


print(np.array(train_points).shape)
print(train_points[0])
print(np.array(train_points[1:3]).shape)

print("Data preprocessing complete\n")


###            DEFINITION OF MODEL SHAPE             ###

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (7, 7), padding='same', activation=tf.nn.leaky_relu,
                               strides=2, input_shape=(512, 512, 1)),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    
    tf.keras.layers.Conv2D(192, (3,3), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    
    tf.keras.layers.Conv2D(128, (1,1), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.Conv2D(256, (3,3), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.Conv2D(256, (1,1), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.Conv2D(512, (3,3), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),

    tf.keras.layers.Conv2D(256, (1,1), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.Conv2D(512, (3,3), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.Conv2D(256, (1,1), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.Conv2D(512, (3,3), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.Conv2D(256, (1,1), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.Conv2D(512, (3,3), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.Conv2D(256, (1,1), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.Conv2D(512, (3,3), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.Conv2D(512, (1,1), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),

    tf.keras.layers.Conv2D(512, (1,1), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.Conv2D(512, (1,1), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.Conv2D(1024, (3,3), padding='same', strides=2, activation=tf.nn.leaky_relu),

    tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.nn.leaky_relu),
    tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.nn.leaky_relu),

    tf.keras.layers.Flatten(), #flatten images into array for the fully connnected layers
#    tf.keras.layers.Dense(1024, activation=tf.nn.leaky_relu),
    #tf.keras.layers.Dropout(0.5), # prevents overfitting for large number of epochs?
#    tf.keras.layers.Dense(4096, activation=tf.keras.activations.linear),
    tf.keras.layers.Dense(4) # 4 outputs: predict 4 points for a bounding box
])
"""
Our final layer predicts both class probabilities and
bounding box coordinates. We normalize the bounding box
width and height by the image width and height so that they
fall between 0 and 1. 

We use a linear activation function for the final layer and
all other layers use the following leaky rectified linear activation:
x if x>0 else 0.1*x
(i think tf.nn.leaky_relu has default of 0.2 instead of 0.1)
"""
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
model.compile(optimizer='adam', 
              loss='mean_squared_error') #TODO: implement YOLO loss(sum-squared error)
             # metrics=['accuracy'])#, options=run_opts)

#print(model.summary())

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
BATCH_SIZE = 1

# reshape training images to expected 4D shape
train_imgs = np.array(train_imgs)
train_points = np.array(train_points)
train_imgs = train_imgs.reshape(-1, 512, 512, 1)

print('making a prediction\n') #try running some test data through the model (untrained) to see if the info it gives back is any more useful for debugging than 
output = model.predict(train_imgs[0:2], batch_size=BATCH_SIZE)
print('out is {}'.format(output))

print('fitting the model\n')
num_epochs = 1
print(np.array(train_points[0:2]).shape)
print(np.array(train_imgs[0:2]).shape)
model.fit(train_imgs, train_points, epochs=num_epochs, batch_size=BATCH_SIZE, \
    steps_per_epoch=(num_train_examples // BATCH_SIZE))

"""
NOTES:
use tf.image.nonMaxSuppression (perform non max sup on of bounding boxes of intersection over the union)

use tf.image.draw_bounding_boxes (draws bb points on images in passed in tensor objects of pts and imgs)

"""

###                 EVALUATE THE MODEL               ###

#TODO: replace train_imgs with testing dataset
loss, accuray = model.evaluate(train_imgs, steps=(num_train_examples // BATCH_SIZE))

