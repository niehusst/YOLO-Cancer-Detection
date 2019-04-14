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

data_frame = pd.read_csv(CSV_PATH)

# zip points data together                        TODO: is this really what we want to do???
train_points = zip(data_frame['start_x'], data_frame['start_y'], \
                       data_frame['end_x'], data_frame['end_y'])
train_img_paths = data_frame['imgPath']

# turn arrays into tf dataset
dataset = tf.data.Dataset.from_tensor_slices((train_points, train_img_paths)) 

# do some preprocessing of the data
def path_to_image(points, path):
    #load image from path as numpy array
    image = pdcm.dcmread(path).pixel_array
    
    return points, image


dataset.map(path_to_image)

# make iterator from dataset
iter = dataset.make_one_shot_iterator()





