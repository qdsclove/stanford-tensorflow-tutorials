from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

import time
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model, Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

from keras.datasets import cifar10
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


# If using TF(NHWC), set image dimensions order to TH (NCHW)
from keras import backend as K
if K.backend() == 'tensorflow':
    K.set_image_dim_ordering("th")

# Use GPU for Theano, comment to use CPU instead of GPU
# Tensorflow uses GPU by default
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
num_train, img_channels, img_rows, img_cols = train_features.shape
num_test, _, _, _ =test_features.shape
num_classes = len(np.unique(train_labels))

print ("Training X:", train_features.shape)
print ("Training Y:", train_labels.shape)
print (test_features.shape)
print (test_labels.shape)
print ("Number of class:", num_classes)


# Data pre-processing
train_X = train_features.astype('float32')/255
test_X = test_features.astype('float32')/255
# convert class labels to one-hot encoding
train_y = np_utils.to_categorical(train_labels, num_classes)
test_y = np_utils.to_categorical(test_labels, num_classes)


# plot model accuracy and loss
def plot_model_history(model_history, tofile='history.png'):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.savefig(tofile)
    #plt.show()


# load base model VGG16
base_model = VGG16(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# add a logistic layer
predictions = Dense(num_classes, activation='softmax')(x)

# final model
model = Model(inputs=base_model.input, outputs=predictions)

# freeze all vgg layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Plot model to image file
plot_model(model, to_file="Model_VGG16_cifar.png", show_shapes=True)

# Set callback functions to early stop training and save the best model so far
early_Stopper = EarlyStopping(monitor='val_acc', patience=50, min_delta=0,mode='auto')
model_check_point = ModelCheckpoint(filepath='Model_VGG16_cifar.h5', monitor='val_acc', save_best_only=True)


# Train the model
start = time.time()
model_info = model.fit(train_X, train_y,
                       batch_size=128, nb_epoch=200,
                       validation_data = (test_X, test_y),
                       verbose=2,
                       callbacks=[early_Stopper, model_check_point])
end = time.time()

# plot model history
plot_model_history(model_info, tofile="cifar_train_history.png")

print ("Model took %0.2f seconds to train"%(end - start))

del model
# load best model
best_model = load_model("Model_VGG16_cifar.h5")

# Evaluate model on test data
score = best_model.evaluate(test_X, test_y, verbose=0)

print ("test loss:", score[0])
print ("test accuracy:", score[1])
