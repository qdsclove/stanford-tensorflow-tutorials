import time
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

from keras.datasets import cifar10
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

# If using TF(NHWC), set image dimensions order to TH (NCHW)
from keras import backend as K
if K.backend() == 'tensorflow':
    K.set_image_dim_ordering("th")

# Use GPU for Theano, comment to use CPU instead of GPU
# Tensorflow uses GPU by default
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

from helper import plot_model_history, write_log_file

(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
num_train, img_channels, img_rows, img_cols = train_features.shape
num_test, _, _, _ = test_features.shape
num_classes = len(np.unique(train_labels))

print ("Training X:", train_features.shape)
print ("Training Y:", train_labels.shape)
print (test_features.shape)
print (test_labels.shape)
print ("Number of class:", num_classes)

# Show some examples
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
fig = plt.figure(figsize=(8,3))
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(train_labels[:]==i)[0]
    features_idx = train_features[idx,::]
    img_num = np.random.randint(features_idx.shape[0])
    im = np.transpose(features_idx[img_num,::], (1, 2, 0))
    ax.set_title(class_names[i])
    plt.imshow(im)
#plt.show()

# Data pre-processing
train_X = train_features.astype('float32')/255
test_X = test_features.astype('float32')/255
# convert class labels to one-hot encoding
train_y = np_utils.to_categorical(train_labels, num_classes)
test_y = np_utils.to_categorical(test_labels, num_classes)


# CNN Model
model = Sequential()
model.add(Conv2D(48, (3, 3), padding='same',input_shape=train_X.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(48, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(96, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(96, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(192, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(192, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

output_dir = "/home/nebula-li/Documents/cs20_data/"
model_image_name = output_dir + "model/1_my_cifar_aug.png"
model_name = output_dir + "weights/1_my_cifar_aug.h5"
training_history_image = output_dir + "log/1_my_cifar_aug.png"
training_result = output_dir + "result/1_my_cifar_aug.log"

# Plot model to image file
plot_model(model, to_file=model_image_name, show_shapes=True)

# Set callback functions to early stop training and save the best model so far
early_Stopper = EarlyStopping(monitor='val_acc', patience=50, min_delta=0,mode='auto')
model_check_point = ModelCheckpoint(filepath=model_name, monitor='val_acc', save_best_only=True)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# data augmentation
datagen_train = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
datagen_train.fit(train_X)


# Train the model
start = time.time()
"""
model_info = model.fit(train_X, train_y,
                       batch_size=128, nb_epoch=300,
                       validation_data = (test_X, test_y),
                       verbose=2,
                       callbacks=[ model_check_point])
"""
model_info = model.fit_generator(datagen_train.flow(train_X, train_y, batch_size=128),
                    validation_data = (test_X, test_y),
                    verbose=2,
                    steps_per_epoch=len(train_X) / 128, epochs=300,
                    callbacks=[model_check_point])

end = time.time()

# plot model history
plot_model_history(model_info, tofile=training_history_image)


print ("Model took %0.2f seconds to train"%(end - start))

del model
# load best model
best_model = load_model(model_name)

# Evaluate model on test data
score = best_model.evaluate(test_X, test_y, verbose=0)

print ("test loss:", score[0])
print ("test accuracy:", score[1])

########## log

min_loss, min_val_loss = min(model_info.history['loss']), min(model_info.history['val_loss'])
max_acc, max_val_acc = max(model_info.history['acc']), max(model_info.history['val_acc'])
# write these values to log file
write_log_file("min_loss", min_loss, tofile=training_result)
write_log_file("min_val_loss", min_val_loss, tofile=training_result)
write_log_file("max_acc", max_acc, tofile=training_result)
write_log_file("max_val_acc", max_val_acc, tofile=training_result)
write_log_file("test loss", score[0], tofile=training_result)
write_log_file("test accuracy", score[1], tofile=training_result)