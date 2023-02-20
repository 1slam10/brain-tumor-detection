import cv2
import os
import random
import numpy as np
from PIL import Image
from keras.models import load_model

IMG_SIZE = 64

model = load_model('BrainTumor20Epochs.h5')

# Getting some random inage from pred folder
number = random.randint(0, 59)
image = cv2.imread(os.path.join('pred', f'pred{number}.jpg'))
image = Image.fromarray(image)
image = image.resize((IMG_SIZE, IMG_SIZE))
image = np.array(image)
image = np.expand_dims(image, axis=0)

result = model.predict(image)

print(number, result)
