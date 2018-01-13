from __future__ import print_function
from os import listdir
from scipy.misc import imread
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import load_model, Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

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

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

model = load_model("model-mnist1 - 1.h5")

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


target_test_image = x_test[1]

plt.imshow(target_test_image.reshape(img_rows, img_cols))
plt.show()

results = model.predict(target_test_image.reshape(1, img_rows, img_cols, 1))
print("predictions with test number - ")
print(results)
print(model.predict_classes(target_test_image.reshape(1, img_rows, img_cols, 1)))

target_test_image = test_data[1]

plt.imshow(target_test_image.reshape(img_rows, img_cols))
plt.show()

results = model.predict(target_test_image.reshape(1, img_rows, img_cols, 1))
print("predictions with test 'a' - ")
print(results)
print(model.predict_classes(target_test_image.reshape(1, img_rows, img_cols, 1)))

# serialize model to JSON
# model_json = model.to_json()
# with open("model-mnist1-1.json", "w") as json_file:
#     json_file.write(model_json)
# serialize weights to HDF5
# model.save_weights("model-mnist1-1.h5")
# print("Saved model to disk")
