import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
from helper.plot_helper import *

batch_size = 128
class_count = 10
epoch_count = 20

(all_train_x, all_train_y), (all_test_x, all_test_y) = mnist.load_data()

def show_digit(x_val):
    show_digit(x_val)

train_x_shape = all_train_x.shape
test_x_shape = all_test_x.shape

all_train_x = all_train_x.reshape(train_x_shape[0], train_x_shape[1] *
        train_x_shape[2])
all_test_x = all_test_x.reshape(test_x_shape[0], test_x_shape[1] *
        test_x_shape[2])

all_train_x = all_train_x.astype('float32')
all_test_x = all_test_x.astype('float32')

all_train_x /= 255
all_test_x /= 255

all_train_y = np_utils.to_categorical(all_train_y, class_count)
all_test_y = np_utils.to_categorical(all_test_y, class_count)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

# We are working with categorical data.
model.compile(loss='categorical_crossentropy', optimizer='RMSprop',
        metrics=['accuracy'])

model.fit(all_train_x, all_train_y, batch_size=batch_size,
        nb_epoch=epoch_count, verbose=1)

score = model.evaluate(all_test_x, all_test_y, verbose=1)

print 'Accuracy: %.2f%%' % (score[1])
