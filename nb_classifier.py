import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from helper.plot_helper import *
from sklearn import cross_validation
import warnings

input_file = 'data/data_multivar.txt'

all_x = []
all_y = []

with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        all_x.append(data[:-1])
        all_y.append(data[-1])

all_x = np.array(all_x)
all_y = np.array(all_y)

#plot_classification_data(all_x, all_y)

classifier_gaussiannb = GaussianNB()

x_train, x_test, y_train, y_test = cross_validation.train_test_split(all_x,
    all_y, test_size=0.25)

classifier_gaussiannb.fit(x_train, y_train)
y_test_pred = classifier_gaussiannb.predict(x_test)

accuracy = 100.0 * (y_test == y_test_pred).sum() / len(y_test)
print 'Accuracy is %.2f' % accuracy

#plot_classifier(classifier_gaussiannb, all_x, all_y)
