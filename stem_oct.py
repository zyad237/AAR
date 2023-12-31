# The "Arabic Alphabet Recognition" project aims to develop an accurate system capable of recognizing characters from the Arabic alphabet.
# This project uses multiple machine learning techniques, including Convolutional Neural Networks (CNNs), 
# to create a model that can accurately identify different Arabic characters from input images. 
# Moreover, tons of information and images were used to achieve the required accuracy of the model.
# Finally, the predictions will be shared in a CSV file. 


import numpy as np
import os
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the path to your dataset
path ="stem-oct-cs-club-1/train/isolated_alphabets_per_alphabet/"

# Define the classes and their corresponding numerical labels
classes = {
    'ain_begin':0,'ain_end':1,'ain_middle':2,'ain_regular':3,
    'alif_end':4,'alif_hamza':5,'alif_regular':6,
    'beh_middle':7,'beh_begin':8,'beh_end':9,'beh_regular':10,
    'dal_end':11,'dal_regular':12,
    'feh_begin':13,'feh_end':14,'feh_middle':15,'feh_regular':16,
    'heh_begin':17,'heh_end':18,'heh_middle':19,'heh_regular':20,
    'jeem_begin':21,'jeem_end':22,'jeem_middle':23,'jeem_regular':24, 
    'kaf_begin':25,'kaf_end':26,'kaf_middle':27,'kaf_regular':28, 
    'lam_alif':29,'lam_begin':30,'lam_end':31,'lam_middle':32,'lam_regular':33, 
    'meem_end':34, 'meem_middle':35,'meem_regular':36,
    'noon_begin':37,'noon_end':38,'noon_middle':39,'noon_regular':40, 
    'qaf_begin':41,'qaf_end':42,'qaf_middle':43,'qaf_regular':44,
    'raa_end':45,'raa_regular':46,
    'sad_begin':47,'sad_end':48,'sad_middle':49,'sad_regular':50,
    'seen_begin':51,'seen_end':52, 'seen_middle':53, 'seen_regular':54, 
    'tah_end':55,'tah_middle':56,'tah_regular':57, 
    'waw_end':58, 'waw_regular':59, 
    'yaa_begin':60,'yaa_end':61,'yaa_middle':62,'yaa_regular':63
}
classes_reversed = {
    0: 'ain_begin', 1: 'ain_end', 2: 'ain_middle', 3: 'ain_regular',
    4: 'alif_end', 5: 'alif_hamza', 6: 'alif_regular',
    7: 'beh_middle', 8: 'beh_begin', 9: 'beh_end', 10: 'beh_regular',
    11: 'dal_end', 12: 'dal_regular',
    13: 'feh_begin', 14: 'feh_end', 15: 'feh_middle', 16: 'feh_regular',
    17: 'heh_begin', 18: 'heh_end', 19: 'heh_middle', 20: 'heh_regular',
    21: 'jeem_begin', 22: 'jeem_end', 23: 'jeem_middle', 24: 'jeem_regular',
    25: 'kaf_begin', 26: 'kaf_end', 27: 'kaf_middle', 28: 'kaf_regular',
    29: 'lam_alif', 30: 'lam_begin', 31: 'lam_end', 32: 'lam_middle', 33: 'lam_regular',
    34: 'meem_end', 35: 'meem_middle', 36: 'meem_regular',
    37: 'noon_begin', 38: 'noon_end', 39: 'noon_middle', 40: 'noon_regular',
    41: 'qaf_begin', 42: 'qaf_end', 43: 'qaf_middle', 44: 'qaf_regular',
    45: 'raa_end', 46: 'raa_regular',
    47: 'sad_begin', 48: 'sad_end', 49: 'sad_middle', 50: 'sad_regular',
    51: 'seen_begin', 52: 'seen_end', 53: 'seen_middle', 54: 'seen_regular',
    55: 'tah_end', 56: 'tah_middle', 57: 'tah_regular',
    58: 'waw_end', 59: 'waw_regular',
    60: 'yaa_begin', 61: 'yaa_end', 62: 'yaa_middle', 63: 'yaa_regular'
}

# Load and preprocess the dataset
data = []
for cl in classes:
    pth = os.path.join(path, cl)
    for img_name in os.listdir(pth):
        img_id = img_name.split('.')[0]
        img = cv2.imread(os.path.join(pth, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        label = classes[cl]
        data.append((img, label, img_id))

data = shuffle(data, random_state=42)
X, Y, image_ids = zip(*data)
X = np.array(X)
Y = np.array(Y)
X = X.reshape(X.shape[0], 64, 64, 1)  # Reshape for Conv2D input

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize pixel values to the range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
accuracy = model.evaluate(X_test, y_test)[1]
print("Accuracy:", accuracy * 100)
