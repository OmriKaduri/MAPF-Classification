import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

print("TensorFlow version is ", tf.__version__)

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_size = 224  # All images will be resized to 160x160
batch_size = 32

# Rescale all images by 1./255 and apply image augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255, validation_split=0.2)

# validation_datagen = keras.preprocessing.image.ImageDataGenerator(
#     rescale=1. / 255)

# Flow training images in batches of 20 using train_datagen generator

train_generator = train_datagen.flow_from_directory(
    '../edgelists/new_format/AllData',  # Source directory for the training images
    target_size=(image_size, image_size),
    batch_size=batch_size,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='categorical',
    subset='training')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = train_datagen.flow_from_directory(
    '../edgelists/new_format/AllData',  # Source directory for the validation images
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

IMG_SHAPE = (image_size, image_size, 3)

#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=IMG_SHAPE))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(5, activation='softmax'))

# Create the base model from the pre-trained model MobileNet V2
# base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
#                                              include_top=False,
#                                              weights='imagenet')
#
# base_model.trainable = False
#
# print(base_model.summary())
#
# model = tf.keras.Sequential([
#     base_model,
#     keras.layers.GlobalAveragePooling2D(),
#     keras.layers.Dense(5, activation='softmax')
# ])

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


print(model.summary())

epochs = 10
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

history = model.fit_generator(train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              workers=4,
                              validation_data=validation_generator,
                              validation_steps=validation_steps)
