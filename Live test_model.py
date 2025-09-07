# import Libraries
import cv2
import numpy as np
import face_recognition
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D

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

#load trianed weights on model
model.load_weights('gender weights.h5')

# Live test
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)

    if ret:
        # Detect all faces in the frame
        face_locations = face_recognition.face_locations(frame)

        for top, right, bottom, left in face_locations:
            # face cut
            face = frame[top:bottom, left:right]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (128,128))
            face = face / 255.0
            face = face.reshape(1,128,128,1)

            # gender prediction with a model
            gender = model.predict(face, verbose=False)

            if gender[0][0] < 0.5:
                label = "Male"
            else:
                label = "Female"

            # Draw a rectangle and text
            cv2.rectangle(frame, (left, top), (right, bottom), (255,0,0), 2)
            cv2.rectangle(frame, (left, bottom), (right, bottom+50), (255,0,0), -1)
            cv2.putText(frame, label, (left+65, bottom+35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 2)

        cv2.imshow('Live', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()