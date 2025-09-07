# import Libraries
import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D
from sklearn.model_selection import train_test_split

#
Base_dir = 'E:/AioLearn - AI/Project/age & gender_detection/UTKFace_dataset'

# preprocessing
images = os.listdir(Base_dir)

image_paths = []
gender_labels = []

for file_name in (images):
    image_path = os.path.join(Base_dir, file_name)
    temp = file_name.split('_')
    if str(temp[1]).isnumeric():
        gender = int(temp[1])
        image_paths.append(image_path)
        gender_labels.append(gender)

def make_dataset(images):
    features = []
    for image_path in images:
        img = load_img(image_path, color_mode='grayscale')
        img = img.resize((128,128), Image.Resampling.LANCZOS)
        img = np.array(img, dtype=np.float32) / 255.0
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features),128,128,1)
    return features

# Data preparation(input X and output Y)
X = make_dataset(image_paths)

Y = np.array(gender_labels)

# Separating data into training and testing
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

# Building a CNN model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(GlobalMaxPooling2D())

model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation= 'sigmoid'))

# compile the CNN model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# training the model
model.fit(x_train,y_train,epochs=10, validation_data=(x_test,y_test), verbose=True)

# save weights model
model.save_weights('gender weights.h5')