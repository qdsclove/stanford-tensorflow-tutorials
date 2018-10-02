import os
import struct


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K

from matplotlib import pyplot as plt
from keras.utils.vis_utils import plot_model

def parse_data(path, dataset, flatten):
    if dataset != 'train' and dataset != 't10k':
        raise NameError('dataset must be train or t10k')

    label_file = os.path.join(path, dataset + '-labels-idx1-ubyte')
    with open(label_file, 'rb') as file:
        _, num = struct.unpack(">II", file.read(8))
        labels = np.fromfile(file, dtype=np.int8)  # int8
        new_labels = np.zeros((num, 10))
        new_labels[np.arange(num), labels] = 1

    img_file = os.path.join(path, dataset + '-images-idx3-ubyte')
    with open(img_file, 'rb') as file:
        _, num, rows, cols = struct.unpack(">IIII", file.read(16))
        imgs = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols)  # uint8
        imgs = imgs.astype(np.float32) / 255.0
        if flatten:
            imgs = imgs.reshape([num, -1])

    return imgs, new_labels


def read_notmnist(path, flatten=True, num_train=55000):
    """
    Read in the mnist dataset, given that the data is stored in path
    Return two tuples of numpy arrays
    ((train_imgs, train_labels), (test_imgs, test_labels))
    """
    imgs, labels = parse_data(path, 'train', flatten)
    indices = np.random.permutation(labels.shape[0])
    train_idx, val_idx = indices[:num_train], indices[num_train:]
    train_img, train_labels = imgs[train_idx, :], labels[train_idx, :]
    val_img, val_labels = imgs[val_idx, :], labels[val_idx, :]
    test = parse_data(path, 't10k', flatten)
    return (train_img, train_labels), (val_img, val_labels), test


def get_notmnist_dataset(batch_size):
    # Step 1: Read in data
    mnist_folder = 'data/notMNIST'

    train, val, test = read_notmnist(mnist_folder, flatten=False)

    # Step 2: Create datasets and iterator
    train_data = tf.data.Dataset.from_tensor_slices(train)
    train_data = train_data.shuffle(10000) # if you want to shuffle your data
    train_data = train_data.batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices(test)
    test_data = test_data.batch(batch_size)

    return train_data, test_data



mnist_folder = 'data/notMNIST'
img_rows, img_cols = 28, 28

(train_img, train_labels), (val_img, val_labels), (test_img, test_labels) = read_notmnist(mnist_folder, flatten=True)

print (train_img.shape)

print (train_labels.shape)
print (val_img.shape)
print (val_labels.shape)
print (test_img.shape)
print (test_labels.shape)

# show something
show_idx = np.random.randint(0, train_img.shape[0])
print ("show_idx:", show_idx)
train_img_show = train_img.reshape(train_img.shape[0], img_rows, img_cols)
print ("image label:", train_labels[show_idx])
plt.imshow(train_img_show[show_idx])
#plt.show()


# reshape data
if K.image_data_format() == 'channels_first':
    x_train = train_img.reshape(train_img.shape[0], 1, img_rows, img_cols)
    x_val = val_img.reshape(val_img.shape[0], 1, img_rows, img_cols)
    x_test = test_img.reshape(test_img.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = train_img.reshape(train_img.shape[0], img_rows, img_cols, 1)
    x_val = val_img.reshape(val_img.shape[0], img_rows, img_cols, 1)
    x_test = test_img.reshape(test_img.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print ("input shape:", input_shape)




# Define model architecture
model = Sequential()

# Input layer
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=input_shape, data_format='channels_last'))
print (model.output_shape)
model.add(Convolution2D(32, (3, 3), activation='relu', padding='valid', data_format='channels_last'))
print (model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
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
early_Stopper = EarlyStopping(monitor='val_loss', patience=10, min_delta=0,mode='auto')

model_check_point = ModelCheckpoint(filepath='best_model_notmnist.h5', monitor='val_loss', save_best_only=True)

# Compile model
model.compile(loss="categorical_crossentropy",
              optimizer='adam',
              metrics=['accuracy']
              )

# Plot model to image file
plot_model(model, to_file="CNN_model.png", show_shapes=True)


model.fit(x_train, train_labels, batch_size=128,
          epochs=100, verbose=2, validation_data=(x_val, val_labels),
          callbacks=[early_Stopper, model_check_point]
          )

# Evaluate model on test data
score = model.evaluate(x_test, test_labels, verbose=1)

print ("test loss:", score[0])
print ("test accuracy:", score[1])



# Predict
y_pred = model.predict(x_test, batch_size=128, verbose=1)

gt_labels = np.argmax(test_labels, axis=1)
predict_labels = np.argmax(y_pred, axis=1)


print (test_labels.shape)
print (y_pred.shape)

wrong_indices = np.where( np.equal(predict_labels, gt_labels) == False)
print ("Total wrong number:", len(wrong_indices[0]))

X_test = x_test.reshape(x_test.shape[0], 28, 28)

for i in range( len(wrong_indices[0])):
    index = wrong_indices[0][i]
    print ("Ground truth:", gt_labels[index], "Predict:", predict_labels[index])

    plt.imshow(X_test[index])
    plt.show()