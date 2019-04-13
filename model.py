# Import tensorflow 
import tensorflow as tf
#import tensorflow_datasets as tfds
tf.logging.set_verbosity(tf.logging.ERROR)

# Helper libraries
import math
import numpy as np
import matplotlib
matplotlib.use('PS') #prevent import error due to venv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Imports for dataset separation
from sklearn.model_selection import train_test_split

# Improve progress bar display
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

# allow for dataset iteration. 
tf.enable_eager_execution() #comment this out if causing errors

###       GET THE DATASET AND SOME INFO ABOUT IT       ###



