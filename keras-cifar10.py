#CNN Model for CIFAR-10 using Keras w/TensorFlow backend.
#80% accuracy or so after ~100 epochs.
#Can be modified to run on Theano backend by changing 'tf' to 'th' in line 15.

import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.constraints import maxnorm
from keras.optimizers import adam
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

#Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#Load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#Normalize inputs 
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

#One hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#Create model
model = Sequential()

model.add(Convolution2D(16, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.3))

model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(Dropout(0.3))

model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.3))

model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))

#Compile model
epochs = 200
lrate = 0.001
decay = lrate/epochs
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#Fit model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=128)

#Final evaluation of model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#Accuracy reported: . Model complete.
