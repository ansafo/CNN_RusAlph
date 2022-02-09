import numpy as np
import os
import random

from PIL import Image
from PIL import ImageFilter

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils

from sklearn.utils import shuffle

# Data loading and augmentation
img_path = r'\PATH_TO_DATASET'
X_train = []
y_train = []
X_test = []
y_test = []

os.chdir(img_path)
dirs = os.listdir()
class_dot = 0
for d in dirs:
    print(d, end=' ')
    index = 0
    os.chdir(os.path.join(img_path, d))
    
    files = os.listdir()
    for f in files:
        for way in ['original' , 'rotate+', 'rotate-', 'blur', 'blur&rotate', 'shift']:
            if way == 'original':
                img = Image.open(f)
                res_img = Image.new("RGB", img.size, (255, 255, 255))
                res_img.paste(img, mask=img.split()[3])
                res_img = res_img.resize((32, 32))
                img_arr = np.array(res_img)
                
            elif way == 'rotate+':
                img = Image.open(f)
                res_img = Image.new("RGB", img.size, (255, 255, 255))
                res_img.paste(img, mask=img.split()[3])
                angle = random.randint(0, 50)
                res_img = res_img.rotate(angle, fillcolor='white')
                res_img = res_img.resize((32, 32))
                img_arr = np.array(res_img)
                
            elif way == 'rotate-':
                img = Image.open(f)
                res_img = res_img.resize((32, 32))
                res_img = Image.new("RGB", img.size, (255, 255, 255))
                res_img.paste(img, mask=img.split()[3])
                angle = random.randint(-50, 0)
                res_img = res_img.rotate(angle, fillcolor='white')
                res_img = res_img.resize((32, 32))
                img_arr = np.array(res_img)
                
            elif way == 'blur':
                img = Image.open(f)
                res_img = Image.new("RGB", img.size, (255, 255, 255))
                res_img.paste(img, mask=img.split()[3])
                res_img = res_img.filter(filter=ImageFilter.GaussianBlur(0.8))
                res_img = res_img.resize((32, 32))
                img_arr = np.array(res_img)
                
            elif way == 'blur&rotate':
                img = Image.open(f)
                res_img = Image.new("RGB", img.size, (255, 255, 255))
                res_img.paste(img, mask=img.split()[3])
                res_img = res_img.filter(filter=ImageFilter.GaussianBlur(0.8))
                angle = random.randint(-50, 0)
                res_img = res_img.rotate(angle, fillcolor='white')
                res_img = res_img.resize((32, 32))
                img_arr = np.array(res_img)
                
            elif way == 'shift':
                img = Image.open(f)
                res_img = Image.new("RGB", img.size, (255, 255, 255))
                res_img.paste(img, mask=img.split()[3])
                res_img = res_img.rotate(0, fillcolor='white')
                horizontal, vertical = random.randint(-5, 5), random.randint(-5, 5)
                res_img = res_img.resize((32, 32))
                img_arr = np.array(res_img)
                
            if index >= round(len(files) / 100 * 85):
                X_test.append(img_arr)
                y_test.append([class_dot])

            else:
                X_train.append(img_arr)
                y_train.append([class_dot])
            
        index += 1
    
    class_dot += 1

X_train = np.array(X_train, dtype='float32')
y_train = np.array(y_train, dtype='uint8')
X_test = np.array(X_test, dtype='float32')
y_test = np.array(y_test, dtype='uint8')

Y_train = np_utils.to_categorical(y_train, 33)
Y_test = np_utils.to_categorical(y_test, 33)

X_train = X_train / 255
X_test, Y_test = shuffle(X_test / 255, Y_test)

print()


# Model architecture
model = Sequential()

model.add(Conv2D(16, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(33, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
history = model.fit(X_train, Y_train,
         batch_size=80,
         epochs=200,
         validation_data=(X_test, Y_test),
         shuffle=True)

os.chdir(r'/PATH_TO_SAVED_MODELS')
filename = ''
model.save_weights(f'{filename}.hdf5')
model.save(f'{filename}.h5')
