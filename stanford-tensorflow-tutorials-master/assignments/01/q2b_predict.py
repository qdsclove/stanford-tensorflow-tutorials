import numpy as np
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt


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



model = load_model("best_model.h5")

# Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)

print ("test loss:", score[0])
print ("test accuracy:", score[1])

# Predict
y_pred = model.predict_classes(X_test, batch_size=128, verbose=1)

print (y_test.shape)
print (y_pred.shape)

wrong_indices = np.where( np.equal(y_test, y_pred) == False)
print ("Total wrong number:", len(wrong_indices[0]))

X_test = X_test.reshape(X_test.shape[0], 28, 28)

for i in range( len(wrong_indices[0])):
    index = wrong_indices[0][i]
    print ("Ground truth:", y_test[index], "Predict:", y_pred[index])

    plt.imshow(X_test[index])
    plt.show()
