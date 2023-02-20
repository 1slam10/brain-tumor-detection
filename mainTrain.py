import cv2
import os
from zipfile import ZipFile
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
import keras.layers as layers
import matplotlib.pyplot as plt

IMG_SIZE = 64

with ZipFile('data.zip', 'r') as f:
    f.extractall()

image_directory = 'datasets'
no_tumor_images = os.listdir(os.path.join(image_directory, 'no'))
yes_tumor_images = os.listdir(os.path.join(image_directory, 'yes'))

no_tumor_dir = os.path.join(image_directory, 'no')
yes_tumor_dir = os.path.join(image_directory, 'yes')

dataset = []
label = []

# preparing dataset
for i, image_name in enumerate(no_tumor_images):
    image = cv2.imread(os.path.join(no_tumor_dir, image_name))
    image = Image.fromarray(image, 'RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    dataset.append(np.array(image))
    label.append(0)


for i, image_name in enumerate(yes_tumor_images):
    image = cv2.imread(os.path.join(yes_tumor_dir, image_name))
    image = Image.fromarray(image, 'RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    dataset.append(np.array(image))
    label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)


# Model building
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), 
                        input_shape=(IMG_SIZE, IMG_SIZE, 3),
                        activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(3, 3)))
model.add(layers.Conv2D(32, (3, 3), kernel_initializer='he_uniform',
                        activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(3, 3)))
model.add(layers.Conv2D(32, (3, 3), kernel_initializer='he_uniform',
                        activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(3, 3)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
            optimizer='adam', metrics=['accuracy'])


history = model.fit(x_train, y_train,
        batch_size=16,
        epochs=20,
        verbose=1,
        validation_data=(x_test, y_test),
        shuffle=False)

model.save('BrainTumor20Epochs.h5')
