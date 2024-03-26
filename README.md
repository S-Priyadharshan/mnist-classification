# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

![image](https://github.com/S-Priyadharshan/mnist-classification/assets/145854138/8a0b8150-2ed8-456d-addf-bfe607d0e03d)


## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Load your google collab page
### STEP 2:
Import the required modules
### STEP 3:
Do the neccessary operation on the dataset and classify it into training data and testing data 
### STEP 4:
Check the dataset to verify its contents
### STEP 5:
Ensure that all the images are in grayscale
### STEP 6:
Code your required Convolution layer model with Max-Pool , Flatten and  Dense layers
### STEP 7:
Compile and fit your model to the given dataset
### STEP 8:
Check the correctness of your model by providing an image of a number and running the model through it

## PROGRAM

### Name: Priyadharshan S
### Register Number: 212223240127


```
import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

imgg=X_train[1]
imgg.shape

plt.imshow(imgg,cmap='gray')

y_train.shape


X_train.min()

X_train.max()

X_train_scaled=X_train/255.0
X_test_scaled=X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[1]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

imgg=X_train[8473]
plt.imshow(imgg,cmap='gray')

y_train[8473]

y_train_onehot[8473]

X_train_scaled.shape

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model=keras.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=10,batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)
metrics.head()

metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))

img = image.load_img('viz.png')


type(img)
plt.imshow(img,cmap='gray')

img_28 = image.load_img('viz.png')
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

print(img_28_gray_scaled.shape)


x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(-1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img = image.load_img('viz.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0


x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)


print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')


img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)


print(x_single_prediction)

  plt.imshow(img_28_gray_inverted_scaled.reshape(28,28),cmap='gray')
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/S-Priyadharshan/mnist-classification/assets/145854138/13d014fe-e973-4275-ae9f-7e2a3c2a00a1)

![image](https://github.com/S-Priyadharshan/mnist-classification/assets/145854138/99c1ffaf-5aff-44ec-8f85-da6f6181493c)

### Classification Report

![Screenshot 2024-03-26 194205](https://github.com/S-Priyadharshan/mnist-classification/assets/145854138/4b1cb779-3c0b-4f81-8184-3b98342eee21)

### Confusion Matrix

![Screenshot 2024-03-26 194249](https://github.com/S-Priyadharshan/mnist-classification/assets/145854138/f02ba5a1-3d7a-4b09-8a86-ff5fd7ff5ffd)

### New Sample Data Prediction
![image](https://github.com/S-Priyadharshan/mnist-classification/assets/145854138/79a0e29b-f9cc-4d65-bd00-698642b6f323)

![Screenshot 2024-03-26 194404](https://github.com/S-Priyadharshan/mnist-classification/assets/145854138/265aff32-ef72-404a-8532-25bd1424f931)

## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
