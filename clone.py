"""
 This is the code for behavior cloning.
 The important setting parameters are 
 - image_save_flg : True (crate new pickle), False (use past pickle)
 - data_mode : normal (160x320), resize (32 x 64)
 - memory_mode : all_in_one (full data on memory), generator (minibatch size)
"""

import csv
import cv2
import numpy as np
import pickle
import math

import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers.core import Dropout, SpatialDropout2D
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import plot_model

# https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py
from keras.layers import merge
from keras.models import Model

import tensorflow as tf


def contrast(image, a):
    """
    Library to change brightness of the image
    """
    lut = [np.uint8(255.0 / (1 + math.exp(-a * (i - 128.) / 255.)))
           for i in range(256)]
    result_image = np.array([lut[value] for value in image.flat],
                            dtype=np.uint8)
    result_image = result_image.reshape(image.shape)
    return result_image


def make_parallel(model, gpu_count):
    """
    Library to use multi gpu by Keras.
    https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py

    not working though...
    """
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat(0, [shape[:1] // parts, shape[1:]])
        stride = tf.concat(0, [shape[:1] // parts, shape[1:]*0])
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    print('debug')
                    print(model.inputs)
                    print(get_slice)
                    print(input_shape)
                    print(i)
                    print(gpu_count)
                    print(x)
                    slice_n = Lambda(get_slice,
                                     output_shape=input_shape,
                                     arguments={'idx': i,
                                                'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)


def normal_model():
    """
    Model Difinition : normal mode
    NOTE : Dropout(rate) in keras and dropout(keep_prob) in tensorflow
           rate: float between 0 and 1. Fraction of the input units to drop.
    # Unlike tensorflow, this is drop ratio! not keep ratio!
    https://github.com/fchollet/keras/blob/master/keras/layers/core.py
    """
    # Model original Size
    model = Sequential()
    #    model.add(Lambda(process, input_shape=shape))
    model.add(Lambda(lambda x: x/255.-0.5, input_shape=(160, 320, 1)))

    # Crop the figure
    model.add(Cropping2D(cropping=((60, 20), (0, 0))))

    # Conv1 Layer - input : (80, 320, 1), output : (40, 160, 24)
    model.add(Conv2D(24, (5, 5),
                     padding="same",
                     subsample=(2, 2),
                     activation="relu"))
    model.add(SpatialDropout2D(0.30))

    # Conv2 Layer - input : (40, 160, 24), output : (20, 80, 36)
    model.add(Conv2D(36, (5, 5),
                     padding="same",
                     subsample=(2, 2),
                     activation="relu"))
    model.add(SpatialDropout2D(0.30))

    # Conv3 Layer - input : (20, 80, 36), output : (10, 40, 48)
    model.add(Conv2D(48, (5, 5),
                     padding="valid",
                     subsample=(2, 2),
                     activation="relu"))
    model.add(SpatialDropout2D(0.30))

    # Conv4 Layer - input : (10, 40, 48), output : (10, 40, 64)
    model.add(Conv2D(64, (3, 3),
                     padding="valid",
                     activation="relu"))
    model.add(SpatialDropout2D(0.30))

    # Conv5 Layer - input : (10, 40, 64), output : (10, 40, 64)
    model.add(Conv2D(64, (3, 3),
                     border_mode="valid",
                     activation="relu"))
    model.add(SpatialDropout2D(0.30))

    # FC0
    model.add(Flatten())
    model.add(Dropout(0.30))

    # FC1
    model.add(Dense(300, activation="relu"))
    model.add(Dropout(0.30))

    # FC2
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.30))

    # FC3
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.30))

    # output
    model.add(Dense(1))

    return model


def resize_model():
    """
    Model Difinition : resized mode
    NOTE : Dropout(rate) in keras and dropout(keep_prob) in tensorflow
           rate: float between 0 and 1. Fraction of the input units to drop.
    # Unlike tensorflow, this is drop ratio! not keep ratio!
    https://github.com/fchollet/keras/blob/master/keras/layers/core.py
    """
    # Model original Size
    model = Sequential()

    # Normalize the data
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(32, 64, 1)))

    # Crop the figure
    model.add(Cropping2D(cropping=((12, 4), (0, 0))))

    # Conv1 Layer - input : (16, 64, 1), output : 8, 32, 24
    model.add(Conv2D(24, (5, 5),
                     padding='same',
                     input_shape=(16, 64, 1)))
    model.add(MaxPooling2D((2, 2)))
#    model.add(Dropout(0.75))
    model.add(Activation('relu'))

    # Conv2 Layer - input : (8, 32, 24), output : 4, 16, 36
    model.add(Conv2D(48, (3, 3),
                     padding='same',
                     input_shape=(8, 32, 24)))
    model.add(MaxPooling2D((2, 2)))
#    model.add(Dropout(0.75))
    model.add(Activation('relu'))

    # Conv3 Layer - input : (4, 16, 48), output : 2, 8, 72
    model.add(Conv2D(72, (3, 3),
                     padding='same',
                     input_shape=(4, 16, 36)))
    model.add(MaxPooling2D((2, 2)))
#    model.add(Dropout(0.75))
    model.add(Activation('relu'))

    # Full Connect Layer - input (2, 8, 72), output : 
    model.add(Flatten())
    model.add(Dense(300))
#    model.add(Dropout(0.75))
    model.add(Activation('relu'))

    model.add(Dense(100))
#    model.add(Dropout(0.75))
    model.add(Activation('relu'))
 
    model.add(Dense(50))
#    model.add(Dropout(0.75))
    model.add(Activation('relu'))

    model.add(Dense(10))
#    model.add(Dropout(0.75))
    model.add(Activation('relu'))

    model.add(Dense(1))

    return model


# Simulation Parameter
image_save_flg = False  # False : use past, True : create new
data_mode = 'normal'  # resize or normal
memory_mode = 'all_in_one'  # 'generator' or 'all_in_one'
edge_min = 40
edge_max = 150


# preprocessing function
def manual_preprocess(image_org):
    image_gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
    # image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    # image_contrast = contrast(image_blue, 10)
    # image_canny = cv2.Canny(image_canny, edge_min, edge_max)
    image_out = image_gray
    return image_out

"""
Read data
"""
lines = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print('data num = ', len(lines))
images = []
measurements = []


"""
Case : 'all_in_one' memory
"""
if memory_mode == 'all_in_one':

    if image_save_flg is False:  # if the data already exists
        print('reading past data from pickle...')
        if data_mode == 'resize':
            images = pickle.load(open("images_resize.p", "rb"))
            measurements = pickle.load(open("measurements_resize.p", "rb"))
        elif data_mode == 'normal':
            images = pickle.load(open("images_normal.p", "rb"))
            measurements = pickle.load(open("measurements_normal.p", "rb"))

    if image_save_flg is True:  # if the data not already exists
        # Data Preprocesing & Data Augmentation
        print('creating new data...')
        for line in lines:

            # each data path
            source_path_center = line[0].split('/')[-1]
            source_path_left = line[1].split('/')[-1]
            source_path_right = line[2].split('/')[-1]
            cur_paths = [source_path_center,
                         source_path_left,
                         source_path_right]

            for path in cur_paths:
                current_path = './IMG/' + path
                image_org = cv2.imread(current_path)

                # preprocessing
                if data_mode == 'resize':
                    image_small = cv2.resize(image_org, (64, 32))
                    image = manual_preprocess(image_small)

                elif data_mode == 'normal':
                    image = manual_preprocess(image_org)

                # add the image
                images.append(image)

                # add the flipped image
                flip_image = cv2.flip(image, 1)
                images.append(flip_image)

            # measurement
            correction = 0.1
            for i in range(len(cur_paths)):
                if i == 0:
                    measurement = float(line[3])
                elif i == 1:
                    measurement = float(line[3]) + correction
                elif i == 2:
                    measurement = float(line[3]) - correction
                measurements.append(measurement)

                # add the flipped label too
                aug_measurement = -1.0 * measurement
                measurements.append(aug_measurement)

        # save the data
        pickle.dump(images, open("images.p", "wb"))
        pickle.dump(measurements, open("measurements.p", "wb"))
    print("Reading 'all_in_memory' done!")

    print('Reshaping data...')
    X_train = np.array(images)
    y_train = np.array(measurements)

    if data_mode == 'resize':
        X_train = X_train.reshape(-1, 32, 64, 1)
    elif data_mode == 'normal':
        X_train = X_train.reshape(-1, 160, 320, 1)

    print(X_train.shape)
    print(y_train.shape)
    print('Reshaping done!')

    """
    Model Selection
    """
    if data_mode == 'resize':
        model = resize_model()
    elif data_mode == 'normal':
        model = normal_model()

    """
    Model Configuration Setting
    """
    print (model.summary())
    # model = make_parallel(model, 2)
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    plot_model(model, to_file='model_now.png')

    """
    Start Training
    """
    hist = model.fit(X_train, y_train,
                     validation_split=0.3,
                     shuffle=True,
                     nb_epoch=20,
                     batch_size=128,
                     verbose=1)

    """
    Show the result
    """
    print(hist.history)
    print("saving model...")
    if data_mode == 'resize':
        model.save('model_resize.h5')
    elif data_mode == 'normal':
        model.save('model_normal.h5')

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('./learning_performance.png', dpi=300)
    plt.show()


"""
Case : 'generator' mode
"""
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                # each data path
                source_path_center = batch_sample[0].split('/')[-1]
                source_path_left = batch_sample[1].split('/')[-1]
                source_path_right = batch_sample[2].split('/')[-1]
                cur_paths = [source_path_center,
                             source_path_left,
                             source_path_right]

                for path in cur_paths:
                    current_path = './IMG/' + path
                    image_org = cv2.imread(current_path)

                    # preprocessing
                    if data_mode == 'resize':
                        image_small = cv2.resize(image_org, (64, 32))
                        image = manual_preprocess(image_small)

                    elif data_mode == 'normal':
                        image = manual_preprocess(image_org)

                    # add the image
                    images.append(image)

                    # add the flipped image
                    flip_image = cv2.flip(image, 1)
                    images.append(flip_image)

                # angles
                correction = 0.1
                for i in range(len(cur_paths)):
                    if i == 0:
                        measurement = float(batch_sample[3])
                    elif i == 1:
                        measurement = float(batch_sample[3]) + correction
                    elif i == 2:
                        measurement = float(batch_sample[3]) - correction
                    angles.append(measurement)

                    # add the flipped label too
                    aug_measurement = -1.0 * measurement
                    angles.append(aug_measurement)

            # trim image to only see section with road
            X_train = np.array(images)
            if data_mode == 'resize':
                X_train = X_train.reshape(-1, 32, 64, 1)
            elif data_mode == 'normal':
                X_train = X_train.reshape(-1, 160, 320, 1)

            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)


if memory_mode == 'generator':
    # compile and train the model using the generator function
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    train_generator = generator(train_samples, batch_size=128)
    validation_generator = generator(validation_samples, batch_size=128)

    """
    Model Selection
    """
    if data_mode == 'resize':
        model = resize_model()
    elif data_mode == 'normal':
        model = normal_model()

    """
    Model Configuration Setting
    """
    print(model.summary())
    print(len(train_samples))
    print(len(validation_samples))
    # model = make_parallel(model, 2)
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    plot_model(model, to_file='model_now.png')

    model.fit_generator(train_generator,
                        samples_per_epoch=len(train_samples),
                        validation_data=validation_generator,
                        nb_val_samples=len(validation_samples),
                        nb_epoch=3)

    """
    Show the result
    """
    print(hist.history)
    print("saving model...")
    if data_mode == 'resize':
        model.save('model_resize_generator.h5')
    elif data_mode == 'normal':
        model.save('model_normal_generator.h5')

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model mean squared error loss (generator)')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('./learning_performance.png', dpi=300)
    plt.show()
