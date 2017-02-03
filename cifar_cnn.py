from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from helper.plot_helper import *

batch_size = 32
classes_count = 10
epoch_count = 200

input_shape = (32, 32)
# RGB
num_channels = 3

(all_train_x, all_train_y), (all_test_x, all_test_y) = cifar10.load_data()

def show_data():
    show_cifar(all_train_x[0])

all_train_y = np_utils.to_categorical(all_train_y, classes_count)
all_test_y = np_utils.to_categorical(all_test_y, classes_count)

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
    input_shape=all_train_x.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(classes_count))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
        metrics=['accuracy'])


all_train_x = all_train_x.astype('float32')
all_test_x = all_test_x.astype('float32')

all_train_x /= 255
all_test_x /= 255

model.fit(all_train_x, all_train_y, batch_size=batch_size,
        nb_epoch=epoch_count, verbose=1)

score = model.evaluate(all_test_x, all_test_y, verbose=1)
print 'Accuracy: %.2f%%' % (score[1])

model.save('models/cifar_cnn.h5')
print 'Model was saved'

