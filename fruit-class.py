import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt



train_path = '/Users/Fujitsu/Desktop/dataset/train'
test_path = '/Users/Fujitsu/Desktop/dataset/test'

BATCH_SIZE = 10

train_batches = ImageDataGenerator(
    preprocessing_function = tf.keras.applications.vgg16.preprocess_input,
    rescale = 1/255.,
    horizontal_flip = True,
    vertical_flip = True,

).flow_from_directory(
    directory = train_path,
    target_size = (20,20),
    classes = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges'],
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    color_mode = 'rgb',                     
)

test_batches = ImageDataGenerator(
    preprocessing_function = tf.keras.applications.vgg16.preprocess_input, rescale = 1/255.
).flow_from_directory(
    directory = test_path,
    target_size = (20,20),
    classes=['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges'],
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    color_mode = 'rgb',
    shuffle = False
)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation=('relu'), input_shape=(20, 20, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation=('relu')))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(128, activation=('relu')))
model.add(Dense(128, activation=('relu')))

model.add(Dense(6, activation=('softmax')))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(test_batches, epochs=20)











