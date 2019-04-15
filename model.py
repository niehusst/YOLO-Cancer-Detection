# Import tensorflow 
import tensorflow as tf
import tensorflow_datasets as tfds
tf.logging.set_verbosity(tf.logging.ERROR)

# Helper libraries
import math
import numpy as np
import matplotlib
matplotlib.use('PS') #prevent import error due to venv
import matplotlib.pyplot as plt
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
tf.enable_eager_execution() #comment this out if causing errors

###       GET THE DATASET AND FIX IT UP A BIT       ###
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

def normalize(img):
    img = img.astype(float)
    img += abs(np.amin(img)) 
    img /= np.amax(img)
    return img

train_imgs = map(path_to_image, train_img_paths) 
train_imgs = map(normalize, train_imgs)

# turn arrays into tf dataset #causes use of >10% system memory???
dataset = tf.data.Dataset.from_tensor_slices((train_points, train_imgs)).repeat().batch(1)

# cast to tf types 
def preprocess(points, img):
    # cast to float values before normalizing
    img = tf.cast(img, tf.float32)
    points = tf.cast(points, tf.float32)
    # resize image
    #img = tf.image.resize_images(img, [448, 448])
    return points, img

dataset = dataset.map(preprocess)

print("Data preprocessing complete\n")

# make iterator from dataset that can be used to train the model
#iter = dataset.make_one_shot_iterator()

## images should be shape 448x448 if we are going to follow YOLO exactly in structure (could use np.resize, but lossy)

### MODEL

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (7, 7), padding='same', activation=tf.nn.relu,
                               input_shape=(512, 512, 1))
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


BATCH_SIZE = 1
dataset = dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)

print('fitting the model\n')
num_epochs = 1
model.fit(dataset, epochs=num_epochs, steps_per_epoch=num_train_examples)



