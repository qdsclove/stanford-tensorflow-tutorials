import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint

from matplotlib import pyplot as plt
from keras.utils.vis_utils import plot_model
#np.random.seed(120)

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print ("Training data size:", X_train.shape)
print ("Test data size:", X_test.shape)

show_idx = np.random.randint(0, X_train.shape[0])
print ("show_idx:", show_idx)
plt.imshow(X_train[show_idx])
#plt.show()

# Reshape the data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

print ("After reshape:", X_train.shape, X_test.shape)

# Convert data types and normalize values
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Define model architecture
model = Sequential()

# Input layer
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1, 28, 28), data_format='channels_first'))
print (model.output_shape)
model.add(Convolution2D(32, (3, 3), activation='relu', data_format='channels_first', padding='valid'))
print (model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Dropout(0.25))
print (model.output_shape)

# Fully connected dense layers
model.add(Flatten())
print ("After Flatten:", model.output_shape)
model.add(Dense(128, activation='relu'))
print ("After Dense(128):", model.output_shape)
model.add(Dropout(0.5))
print ("After Dropout:", model.output_shape)
model.add(Dense(10, activation='softmax'))
print ("After Dense(10):", model.output_shape)

# Set callback functions to early stop training and save the best model so far
early_Stopper = EarlyStopping(monitor='val_loss', patience=5, min_delta=0,mode='auto')

model_check_point = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)

callbacks = [EarlyStopping(monitor='val_loss', patience=2, min_delta=0),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

# Compile model
model.compile(loss="categorical_crossentropy",
              optimizer='adam',
              metrics=['accuracy']
              )

# Plot model to image file
plot_model(model, to_file="CNN_model.png", show_shapes=True)


model.fit(X_train, Y_train, batch_size=128,
          epochs=10, verbose=1, validation_data=(X_test, Y_test),
          callbacks=[early_Stopper, model_check_point]
          )

# Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)

print ("test loss:", score[0])
print ("test accuracy:", score[1])

# Predict
y_pred = model.predict_classes(X_test, batch_size=128, verbose=1)
