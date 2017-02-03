from helper.character_table import CharacterTable
from keras.models import load_model
from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
import numpy as np
from helper.character_table import gen_numbers
from sklearn import cross_validation

def show_questions():
    print '%s = %s' % (questions[0], expected[0])


def get_encoded(digits, max_length, chars, ctable, questions, expected):
    X = np.zeros((len(questions), max_length, len(chars)), dtype=np.bool)
    y = np.zeros((len(questions), digits + 1, len(chars)), dtype=np.bool)

    for i, sentence in enumerate(questions):
        X[i] = ctable.encode(sentence, max_length)

    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, digits + 1)

    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    return (X,y)


def build_model(digits, max_length, chars, ctable):
    questions, expected = gen_numbers(5000, digits)

    print 'Encoding data'
    X, y = get_encoded(digits, max_length, chars, ctable, questions, expected)

    print 'Splitting data'
    X_train, X_val, y_train, y_val = cross_validation.train_test_split(X,
        y, test_size=0.1)

    RNN = recurrent.LSTM
    HIDDEN_SIZE = 128
    batch_size = 128
    epoch_count = 100

    print 'Creating network'
    model = Sequential()

    model.add(RNN(HIDDEN_SIZE, input_shape=(max_length, len(chars))))

    # Repeats the input N times
    model.add(RepeatVector(digits + 1))

    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

    model.add(TimeDistributed(Dense(len(chars))))

    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epoch_count,
            verbose=1)

    model.save('models/rnn_add.h5')

    score = model.evaluate(X_val, y_val, verbose=1)

    print 'Accuracy: %.2f%%' % (score[1])

    return model


digits = 2
chars = ' 0123456789+'
max_length = digits + 1 + digits
ctable = CharacterTable(chars)

try:
    model = load_model('models/rnn_add.h5')
    print 'Model loaded'
except IOError:
    model = build_model(digits, max_length, chars, ctable)
    print 'Model built'

while True:
    print '(q to quit)'
    digit1 = raw_input('Enter digit 1: ')
    if digit1 == 'q':
        break

    digit2 = raw_input('Enter digit 2: ')

    if len(digit1) > 3 or len(digit1) <= 0 or len(digit2) > 3 or len(digit2) <= 0:
        print 'Invalid number'
        continue

    #while len(digit1) != 3:
    #    digit1 = ' ' + digit1
    #while len(digit2) != 3:
    #    digit2 = ' ' + digit2

    question = digit1 + '+' + digit2
    question += ' ' * (max_length - len(question))

    input_x = ctable.encode(question, max_length)

    print 'Predicting %s' % (question)
    print '\n' * 2
    print input_x
    print '\n' * 2

    input_x = [input_x]

    input_x = np.array(input_x)

    encoded_pred = model.predict_classes([input_x])
    pred = ctable.decode(encoded_pred[0], calc_argmax=False)

    print pred

