# Note about the dataset.
# 1. Number of times pregnant
# 2. Plasma glucose concentration a 2 hours
#    in an oral glucose tolerance test
# 3. Diastolic blood pressure (mm Hg)
# 4. Triceps skin fold thickness (mm)
# 5. 2-Hour serum insulin (mu U/ml)
# 6. Body mass index (weight in kg/(height in m)^2)
# 7. Diabetes pedigree function
# 8. Age (years)
# 9. Class variable (0 or 1)

import numpy as np
from sklearn import cross_validation
from keras.models import Sequential
from keras.layers import Dense

dataset = np.loadtxt('data/pima_indians_diabetes.csv', delimiter=',')

all_x = [dataset_point[0:8] for dataset_point in dataset]
all_y = [dataset_point[-1] for dataset_point in dataset]

all_x = np.array(all_x)
all_y = np.array(all_y)

x_train, x_test, y_train, y_test = cross_validation.train_test_split(all_x,
        all_y, test_size=0.2)

# Now let's create our actual model.
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# We are working with binary data
model.compile(loss='binary_crossentropy', optimizer='SGD',
    metrics=['accuracy'])

model.fit(x_train, y_train, nb_epoch=150, batch_size=10, verbose=True)

scores = model.evaluate(x_test, y_test, verbose=True)

print ''

for i in range(len(scores)):
    print '%s : %.2f' % (model.metrics_names[i], scores[i])
