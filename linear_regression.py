import sys
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import sklearn.metrics as sm

filename = 'data/data_singlevar.txt'

all_x = []
all_y = []

with open(filename, 'r') as f:
    for line in f.readlines():
        x, y = [float(i) for i in line.split(',')]
        all_x.append(x)
        all_y.append(y)

training_count = int(0.8 * len(all_x))
test_count = len(all_x) - training_count

x_train = np.array(all_x[:training_count]).reshape((training_count, 1))
y_train = np.array(all_y[:training_count])

x_test = np.array(all_x[training_count:]).reshape((test_count, 1))
y_test = np.array(all_y[training_count:])

linear_regressor = linear_model.LinearRegression()

linear_regressor.fit(x_train, y_train)

y_test_pred = linear_regressor.predict(x_test)

plt.scatter(x_test, y_test, color='green')
plt.plot(x_test, y_test_pred, color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.savefig('output/output.png')

print 'Mean Squared Error=%.2f' % (sm.mean_squared_error(y_test, y_test_pred))

