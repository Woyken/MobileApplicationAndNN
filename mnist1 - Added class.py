from __future__ import print_function
from os import listdir
from scipy.misc import imread
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.utils import shuffle

batch_size = 128
num_classes = 11
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

imagelist = []
for item in listdir("EmnistData\\mnist\\train-images2"):
    imagelist.append(imread("EmnistData\\mnist\\train-images2\\" + item))
train_data = np.array(imagelist[:3600])
test_data = np.array(imagelist[3600:])
train_labels = np.full(3600, 10, dtype=int)
test_labels = np.full(1200, 10, dtype=int)

train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = np.concatenate((x_train, train_data))
y_train = np.concatenate((y_train, train_labels))

x_train, y_train = shuffle(x_train, y_train)

x_test = np.concatenate((x_test, test_data))
y_test = np.concatenate((y_test, test_labels))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          shuffle=True)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#results = model.predict(imread("EmnistData\\mnist\\train-images2\\124592.jpg").reshape(1, 28, 28, 1))
results = model.predict(test_data[0])
print("predictions with test 'a' - ")
print(results)

# serialize model to JSON
# model_json = model.to_json()
# with open("model-mnist1-1.json", "w") as json_file:
#     json_file.write(model_json)
# serialize weights to HDF5
# model.save_weights("model-mnist1-1.h5")
# print("Saved model to disk")

model.save("model-mnist1 - 1.h5")
print('saved model to file')
