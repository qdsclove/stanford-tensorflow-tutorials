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

# compute test accuracy
def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class)
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy*100)

# CNN Model
model = Sequential()
model.add(Conv2D(48, (3, 3), padding='same',input_shape=train_X.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(48, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(96, (3, 3), border_mode='same'))
model.add(Activation('relu'))

model.add(Conv2D(96, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(192, (3, 3), border_mode='same'))
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


# Plot model to image file
plot_model(model, to_file="CNN_cifar_model.png", show_shapes=True)

# Set callback functions to early stop training and save the best model so far
early_Stopper = EarlyStopping(monitor='val_loss', patience=50, min_delta=0,mode='auto')
model_check_point = ModelCheckpoint(filepath='best_model_cifar.h5', monitor='val_loss', save_best_only=True)
checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
start = time.time()
model_info = model.fit(train_X, train_y,
                       batch_size=128, nb_epoch=200,
                       validation_data = (test_X, test_y),
                       verbose=2,
                       callbacks=[ model_check_point])
end = time.time()

# plot model history
plot_model_history(model_info, tofile="cifar_train_history.png")

print ("Model took %0.2f seconds to train"%(end - start))

del model
# load best model
best_model = load_model("best_model_cifar.h5")

# compute test accuracy
print ("Accuracy on test data is: %0.2f"%accuracy(test_X, test_y, best_model))

# Evaluate model on test data
score = best_model.evaluate(test_X, test_y, verbose=0)

print ("test loss:", score[0])
print ("test accuracy:", score[1])