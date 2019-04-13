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
# relative paths to data
CSV_PATH = 'CCC_clean.csv'
IMAGE_PATH = '../data/'

dataset = tf.contrib.data.make_csv_dataset(CSV_PATH, batch_size=32)
iter = dataset.make_one_shot_iterator()
next = iter.next()

# how to load an image
rel_path = '/home/niehusst/vision262/project/TCGA-09-0364/1.3.6.1.4.1.14519.5.2.1.7695.4007.115512319570807352125051359179/42'
matrix = pdcm.dcmread(rel_path).pixel_array


