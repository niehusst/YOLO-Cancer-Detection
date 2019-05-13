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
import time

# Imports for dataset manipulation
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# Improve progress bar display
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

#tf.enable_eager_execution() #comment this out if causing errors
tf.logging.set_verbosity(tf.logging.DEBUG)



###             SET MODEL CONFIGURATIONS             ###
# Data Loading
CSV_PATH = 'label_data/CCC_clean.csv'
IMAGE_BASE_PATH = '../data/'
test_size_percent = 0.15 #percent of total data reserved for testing

# Data Augmentation
mirror_im = False

# Loss
lambda_coord = 5

# Learning
step_size = 0.00001
BATCH_SIZE = 5
num_epochs = 5

# Saving
shape_path = 'trained_model/model_shape.json'
weight_path = 'trained_model/model_weights.h5'

# TensorBoard 
tb_graph = False
tb_update_freq = 'batch'
#tb_write_gradients = True
#tb_histogram_freq = 2


            
###         GET THE DATASET AND PREPROCESS IT        ###

print("Loading and processing data\n") #TODO: maybe elimanate non-radiologist data? maybe that's killing model

data_frame = pd.read_csv(CSV_PATH)

"""
Construct numpy ndarrays from the loaded csv to use as training
and testing datasets.
""" 
# zip all points for each image label together into a tuple 
points = zip(data_frame['start_x'], data_frame['start_y'], \
                       data_frame['end_x'], data_frame['end_y'])
img_paths = data_frame['imgPath']


def path_to_image(path):
    """
    Load a matrix of pixel values from the DICOM image stored at the
    input path.
    
    @param path - string, relative path (from IMAGE_BASE_PATH) to
                  a DICOM file
    @return image - numpy ndarray (int), 2D matrix of pixel
                    values of the image loaded from path
    """
    #load image from path as numpy array
    image = pydicom.dcmread(os.path.join(IMAGE_BASE_PATH, path)).pixel_array
    return image

# normalize dicom image pixel values to 0-1 range
def normalize_image(img):
    """
    Normalize the pixel values in img to be withing the range
    of 0 to 1.
    
    @param img - numpy ndarray, 2D matrix of pixel values
    @return img - numpy ndarray (float), 2D matrix of pixel values, every
                  element is valued between 0 and 1 (inclusive)
    """
    img = img.astype(np.float32)
    img += abs(np.amin(img)) #account for negatives
    img /= np.amax(img)
    return img

# normalize the ground truth bounding box labels wrt image dimensions
def normalize_points(points):
    """
    Normalize values in points to be within the range of 0 to 1.
    
    @param points - 1x4 tuple, elements valued in the range of 0
                    512 (inclusive). This is known from the nature
                    of the dataset used in this program
    @return - 1x4 numpy ndarray (float), elements valued in range
              0 to 1 (inclusive)
    """
    imDims = 512.0 #each image is 512x512
    points = list(points)
    for i in range(len(points)):
        points[i] /= imDims
    return np.array(points).astype(np.float32)

"""
Convert the numpy array of paths to the DICOM images to pixel 
matrices that have been normalized to a 0-1 range.
Also normalize the bounding box labels to make it easier for
the model to predict on them.
"""
# apply preprocessing functions
points = map(normalize_points, points)
imgs = map(path_to_image, img_paths)
#imgs = map(normalize_image, imgs)

# reshape input image data to 4D shape (as expected by the model)
# and cast all data to np arrays (just in case)
imgs = np.array(imgs)
points = np.array(points)
imgs = imgs.reshape(-1, 512, 512, 1)

# split the preprocessed data into train and test
train_imgs, test_imgs, train_points, test_points = \
  train_test_split(imgs, points, test_size=test_size_percent, random_state=42)

num_train_examples = len(train_imgs)
num_test_examples = len(test_imgs)


"""
Create generator for feeding the training data to the model
in batches. This specific generator is also capable of data
augmentation.
"""
generator = ImageDataGenerator(rotation_range=0, zoom_range=0,
	width_shift_range=0, height_shift_range=0, shear_range=0,
	horizontal_flip=mirror_im, fill_mode="nearest")
#TODO: use data augment to flip? (change steps per batch so all images get seen in an epoch!) also the labels would need to be editted to be flipped as well????

print("Data preprocessing complete\n")



###            DEFINITION OF MODEL SHAPE             ###
"""
Model definition according (approximately) to the YOLO model 
described by Redmon et. al. in "You Only Look Once:
Unified, Real-Time Object Detection"
"""
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
    tf.keras.layers.Dense(1024),
    #tf.keras.layers.Dropout(0.5), # prevents overfitting for large number of epochs?
    tf.keras.layers.Dense(4096, activation=tf.keras.activations.linear),
    tf.keras.layers.Dense(4, activation=tf.nn.tanh) # 4 outputs: predict 4 points for a bounding box
])
#TODO: try tanh activation instead of sigmoid
#TODO: add tf.keras.layers.BatchNormalization layers before max pooling layers?

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

leaky_relu gets us worse accuracy than regular relu
"""

# custom loss function using aspects of relevant information from the YOLO paper
def YOLO_loss(y_true, y_pred):
    
    x_LT = y_true[:, 0]
    y_UT = y_true[:, 1]
    x_RT = y_true[:, 2]
    y_LT = y_true[:, 3]

    x_LP = y_pred[:, 0]
    y_UP = y_pred[:, 1]
    x_RP = y_pred[:, 2]
    y_LP = y_pred[:, 3]


    x_Pmid = tf.math.add(x_LP, tf.math.divide(tf.math.subtract(x_RP, x_LP), 2))
    x_Tmid = tf.math.add(x_LT, tf.math.divide(tf.math.subtract(x_RT, x_LT), 2))
    y_Pmid = tf.math.add(y_UP, tf.math.divide(tf.math.subtract(y_LP, y_UP), 2))
    y_Tmid = tf.math.add(y_UT, tf.math.divide(tf.math.subtract(y_LT, y_UT), 2))

    x_mid_sqdiff = tf.math.square(tf.math.subtract(x_Pmid, x_Tmid))
    y_mid_sqdiff = tf.math.square(tf.math.subtract(y_Pmid, y_Tmid))
    first_term = tf.math.add(x_mid_sqdiff, y_mid_sqdiff)

    x_Pwidth = tf.math.sqrt(tf.math.abs(tf.math.subtract(x_RP, x_LP)))
    x_Twidth = tf.math.sqrt(tf.math.abs(tf.math.subtract(x_RT, x_LT)))
    y_Pheight = tf.math.sqrt(tf.math.abs(tf.math.subtract(y_UP, y_LP)))
    y_Theight = tf.math.sqrt(tf.math.abs(tf.math.subtract(y_UT, y_LT)))



    second_term = tf.math.add(tf.math.square(tf.math.subtract(x_Pwidth,  x_Twidth)),
                              tf.math.square(tf.math.subtract(y_Pheight, y_Theight)))

    loss = tf.math.multiply(tf.math.add(first_term, second_term), lambda_coord)
    return loss #+ tf.keras.losses.mean_squared_error(y_true, y_pred)

def IOU_metric(y_true, y_pred):
    """
    Compute the simple, straightforward intersection over the union
    of the true and predicted bounding boxes. Output in range 0-1, 
    1 being the best match of bounding boxes (perfect alignment), 
    0 being worst (no intersection at all).
    
    @param y_true - BATCH_SIZEx4 Tensor object (float), the ground 
                    truth labels for the bounding boxes around the
                    tumor in the corresponding images. Value order:
                    (x_top_left, y_top_left, x_bot_right, y_bot_right)
    @param y_pred - BATCH_SIZEx4 Tensor object (float), the model's 
                    prediction for the bounding boxes around the
                    tumor in the corresponding images. Value order:
                    (x_top_left, y_top_left, x_bot_right, y_bot_right)
    @return iou - 1x1 Tensor object (float), value being the mean
                  IOU for all image in the batch, and is within the 
                  range of 0-1 (inclusive). 
    """
    # extract points from tensors
    x_LT = tf.math.minimum(y_true[:, 0], y_true[:, 2])
    y_UT = tf.math.minimum(y_true[:, 1], y_true[:, 3])
    x_RT = tf.math.maximum(y_true[:, 0], y_true[:, 2])
    y_LT = tf.math.maximum(y_true[:, 1], y_true[:, 3])

    x_LP = tf.math.minimum(y_pred[:, 0], y_pred[:, 2])
    y_UP = tf.math.minimum(y_pred[:, 1], y_pred[:, 3])
    x_RP = tf.math.maximum(y_pred[:, 0], y_pred[:, 2])
    y_LP = tf.math.maximum(y_pred[:, 1], y_pred[:, 3])


    xL_pairwise_gt = K.greater(x_LT, x_LP)
    yU_pairwise_gt = K.greater(y_UT, y_UP)

    xW1_pairwise_int = K.less(x_LT, x_RP)
    xW1 = K.cast(xW1_pairwise_int, K.floatx())

    xW2_pairwise_int = K.less(x_LP, x_RT)
    xW2 = K.cast(xW2_pairwise_int, K.floatx())

    yH1_pairwise_int = K.less(y_UT, y_LP)
    yH1 = K.cast(yH1_pairwise_int, K.floatx())

    yH2_pairwise_int = K.less(y_UP, y_LT)
    yH2 = K.cast(yH2_pairwise_int, K.floatx())

    x_bin = K.cast(xL_pairwise_gt, K.floatx())
    y_bin = K.cast(yU_pairwise_gt, K.floatx())

    x_does_intersect   = tf.math.add(tf.math.multiply(x_bin, xW1),
                                     tf.math.multiply(tf.math.subtract(1.0, x_bin), xW2))
    y_does_intersect   = tf.math.add(tf.math.multiply(y_bin, yH1),
                                     tf.math.multiply(tf.math.subtract(1.0, y_bin), yH2))
    box_does_intersect = tf.math.multiply(x_does_intersect, y_does_intersect)

    a = tf.math.minimum(tf.math.subtract(x_RP, x_LT), tf.math.subtract(x_RP, x_LP))
    b = tf.math.minimum(tf.math.subtract(x_RT, x_LP), tf.math.subtract(x_RT, x_LT))
    c = tf.math.minimum(tf.math.subtract(y_LP, y_UT), tf.math.subtract(y_LP, y_UP))
    d = tf.math.minimum(tf.math.subtract(y_LT, y_UP), tf.math.subtract(y_LT, y_UT))

    # calculate intersection area
    intersection_width  = tf.math.add(tf.math.multiply(x_bin, a),
                                      tf.math.multiply(tf.math.subtract(1.0, x_bin), b))
    intersection_height = tf.math.add(tf.math.multiply(y_bin, c),
                                      tf.math.multiply(tf.math.subtract(1.0, y_bin), d))

    intersection = tf.math.multiply(tf.math.multiply(intersection_width, intersection_height), box_does_intersect)
    # calculate union area
    union_double = tf.math.add(tf.math.multiply(tf.math.subtract(x_RP, x_LP), tf.math.subtract(y_LP,y_UP)),
                               tf.math.multiply(tf.math.subtract(x_RT, x_LT), tf.math.subtract(y_LT, y_UT)))
    union = tf.math.subtract(union_double, intersection)

    # take the mean in order to compress BATCH_SIZEx1 Tensor
    # into a 1x1 Tensor
    iou = K.mean(tf.math.divide(intersection, union))
    return iou



# small step size works best
model.compile(optimizer=tf.keras.optimizers.SGD(lr=step_size),
              loss=YOLO_loss,
              metrics=['accuracy', IOU_metric, 'mse'])

#print(model.summary()) #see the shape of the model


###                   TRAIN THE MODEL                ###
"""
From the paper:
We train the network for about 135 epochs(thats a lot, they required dropout and data aug).
Throughout training we use a batch size of 64, a momentum of 0.9 and a decay of 0.0005.

Our  learning  rate  schedule  is  as  follows:  For  the  first epochs
we slowly raise the learning rate from 10e-3 to 10e-2. If we start at a
high learning rate our model often diverges due to unstable gradients.
We continue training with 10e-2 for 75 epochs, then 10e-3 for 30 epochs,
and finally 10e-4 for 30 epochs
"""
callbacks = []

# use tensorboard to visualize training on localhost:6006. Call from terminal with:
#>tensorboard --logdir=/full/path/to/logs
callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(time.time()),
                                                 write_graph=tb_graph,
                                                 batch_size=BATCH_SIZE,
                                                 update_freq=tb_update_freq))
""", #histogram visualizing broken
                                                 histogram_freq=tb_histogram_freq,
                                                 write_grads=tb_write_gradients))
"""

print('Fitting the model\n')
# start training the model using the data generator and the configurations
# specified at the top of the file
model.fit_generator(generator.flow(train_imgs, train_points, batch_size=BATCH_SIZE),
                        callbacks=callbacks, epochs=num_epochs,
                        steps_per_epoch=(num_train_examples // BATCH_SIZE))



###                 EVALUATE THE MODEL               ###

# evaluate the accuracy of the trained model using the test dataset
metrics = model.evaluate(test_imgs, test_points)
print("Final loss:{}\nFinal accuracy:{}".format(metrics[0], metrics[1]))



###                 SAVING THE MODEL                 ###

# save the model so that it can be loaded without training later
# serialize model to JSON
model_json = model.to_json()
with open(shape_path, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(weight_path)
print("Saved model to disk")

