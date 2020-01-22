import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(50,50,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.2))


model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.2))

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(units=512,activation='relu'))

model.add(Dense(units=2,activation='softmax'))
model.summary()

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import np_utils

os.chdir('E:\\Locker\\Sai\\SaiHCourseNait\\DecBtch\\R_Datasets\\')
infected_data = os.listdir('cell-images-for-detecting-malaria\\cell_images\\Parasitized\\')
print(infected_data[:10]) 

uninfected_data = os.listdir('cell-images-for-detecting-malaria\\cell_images\\Uninfected\\')
print('\n')
print(uninfected_data[:10])

data = []
labels = []
for img in infected_data:
    try:
        img_read = plt.imread('cell-images-for-detecting-malaria\\cell_images\\Parasitized/' + "/" + img)
        img_resize = cv2.resize(img_read, (50, 50))
        img_array = img_to_array(img_resize)
        img_aray=img_array/255
        data.append(img_array)
        labels.append(1)
    except:
            None

for img in uninfected_data:
    try:
        img_read = plt.imread('cell-images-for-detecting-malaria\\cell_images\\Uninfected' + "/" + img)
        img_resize = cv2.resize(img_read, (50, 50))
        img_array = img_to_array(img_resize)
        
        img_array= img_array/255
        data.append(img_array)
        labels.append(0)
    except:
        None

data[:10]

labels[:10]

plt.imshow(data[0])
plt.show()

image_data = np.array(data)
labels = np.array(labels)
idx = np.arange(image_data.shape[0])
print(idx)

np.random.shuffle(idx)
image_data = image_data[idx]
print(len(image_data))
labels = labels[idx]

print(labels[:10])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(image_data, labels, test_size = 0.2, random_state = 42)

y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)

print(f'Shape of training image : {x_train.shape}')
print(f'Shape of testing image : {x_test.shape}')
print(f'Shape of training labels : {y_train.shape}')
print(f'Shape of testing labels : {y_test.shape}')

H = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=110) 












