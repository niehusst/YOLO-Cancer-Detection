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
#from keras.callbacks import TensorBoard

#tf.enable_eager_execution() #comment this out if causing errors
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

leaky_relu gets us worse accuracy than regular relu
"""

# custom loss function using aspects of relevant information from the YOLO paper
def YOLO_loss(y_true, y_pred):

    # extract points from tensors
    x_LT = tf.math.minimum(y_true[:, 0], y_true[:, 2])
    y_UT = tf.math.minimum(y_true[:, 1], y_true[:, 3])
    x_RT = tf.math.maximum(y_true[:, 0], y_true[:, 2])
    y_LT = tf.math.maximum(y_true[:, 1], y_true[:, 3])

    x_LP = tf.math.minimum(y_pred[:, 0], y_pred[:, 2])
    y_UP = tf.math.minimum(y_pred[:, 1], y_pred[:, 3])
    x_RP = tf.math.maximum(y_pred[:, 0], y_pred[:, 2])
    y_LP = tf.math.maximum(y_pred[:, 1], y_pred[:, 3])

    lambda_coord = 5

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


    intersection_width  = tf.math.add(tf.math.multiply(x_bin, a),
                                      tf.math.multiply(tf.math.subtract(1.0, x_bin), b))
    intersection_height = tf.math.add(tf.math.multiply(y_bin, c),
                                      tf.math.multiply(tf.math.subtract(1.0, y_bin), d))

    intersection = tf.math.multiply(tf.math.multiply(intersection_width, intersection_height), box_does_intersect)
    union_double = tf.math.add(tf.math.multiply(tf.math.subtract(x_RP, x_LP), tf.math.subtract(y_LP,y_UP)),
                               tf.math.multiply(tf.math.subtract(x_RT, x_LT), tf.math.subtract(y_LT, y_UT)))
    union = tf.math.subtract(union_double, intersection)
    iou = K.mean(tf.math.divide(intersection, union))

    loss = tf.math.add(tf.math.multiply(tf.math.add(first_term, second_term), lambda_coord), iou)

    return iou

def IOU_metric(y_true, y_pred):
    x_LT = y_true[:, 0] # left x coord
    y_UT = y_true[:, 1] # upper y coord
    x_RT = y_true[:, 2] # right x coord
    y_LT = y_true[:, 3] # lower y coord
    # predicted points
    x_LP = y_pred[:, 0] # left x coord
    y_UP = y_pred[:, 1] # upper y coord
    x_RP = y_pred[:, 2] # right x coord
    y_LP = y_pred[:, 3] # lower y coord
    # calculate the intersection over the union
    intersection = tf.math.multiply(tf.math.abs(tf.math.subtract(x_LT, x_RP)),
                                    tf.math.abs(tf.math.subtract(y_UT, y_LP)))
    union_double = tf.math.add(tf.math.multiply(tf.math.subtract(x_RP, x_LP),
                tf.math.subtract(y_LP,y_UP)),tf.math.multiply(tf.math.subtract(x_RT, x_LT),
                          tf.math.subtract(y_LT, y_UT)))
    union = tf.math.abs(tf.math.subtract(union_double, intersection))
    #iou = tf.math.divide(intersection, union) #TODO: IOU actually increases as match is better; it should decrease
    #invert true iou equation and add epsilon to intersection to prevent divide-by-0 error
    epsilon = 0.00001
    iou = tf.math.divide(union, tf.math.add(intersection, epsilon))
    return iou

#TODO: adjust parameters for adam optimizer; change learning rate?
model.compile(optimizer='adam',
              loss=YOLO_loss,
              metrics=['accuracy', IOU_metric])

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
num_epochs = 5
callbacks = []

# train with tensorboard to visualize training on localhost:6006. Call from terminal with:
#tensorboard --logdir=/full/path/to/logs
callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(time.time()),
                                                 write_graph=False,
                                                 batch_size=BATCH_SIZE,
                                                 update_freq='batch'))
# save model at checkpoints
callbacks.append(tf.keras.callbacks.ModelCheckpoint('checkpoints/best_weights.h5',
                                    monitor='val_loss', verbose=0, save_best_only=True,
                                    save_weights_only=True, mode='auto', period=1))

print('Fitting the model\n')
model.fit_generator(generator.flow(train_imgs, train_points, batch_size=BATCH_SIZE),
                        callbacks=callbacks, epochs=num_epochs,
                        steps_per_epoch=(num_train_examples // BATCH_SIZE))



###                 EVALUATE THE MODEL               ###

# evaluate the accuracy of the trained model using the test datasets
loss, accuracy = model.evaluate(test_imgs, test_points)
print("Final loss:{}\nFinal accuracy:{}".format(loss, accuracy))



###                 SAVING THE MODEL                 ###

# save the model so that it can be loaded without training later
shape_path = 'trained_model/model_shape.json'
weight_path = 'trained_model/model_weights.h5'

# serialize model to JSON
model_json = model.to_json()
with open(shape_path, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(weight_path)
print("Saved model to disk")

