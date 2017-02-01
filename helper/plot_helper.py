import numpy as np
import matplotlib.pyplot as plt

def plot_classification_data(X, y):
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0

    # denotes the step size that will be used in the mesh grid
    step_size = 0.01

    # define the mesh grid
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size),
        np.arange(y_min, y_max, step_size))

    # Plot the output using a
    # colored plot
    plt.figure()

    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black',
            linewidth=1, cmap=plt.cm.Paired)

    plt.xlim(x_values.min(),
            x_values.max())
    plt.ylim(y_values.min(),
            y_values.max())

    plt.xticks((np.arange(int(min(X[:, 0])-1), int(max(X[:, 0])+1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1])-1), int(max(X[:, 1])+1), 1.0)))

    plt.savefig('output/output.png')


def plot_classifier(classifier, X, y):
    # define ranges to plot the figure
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0

    # denotes the step size that will be used in the mesh grid
    step_size = 0.01

    # define the mesh grid
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size),
        np.arange(y_min, y_max, step_size))

    # compute the classifier output
    mesh_output = classifier.predict(np.c_[x_values.ravel(),
        y_values.ravel()])

    # reshape the array
    mesh_output = mesh_output.reshape(x_values.shape)

    # Plot the output using a
    # colored plot
    plt.figure()

    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black',
            linewidth=1, cmap=plt.cm.Paired)

    plt.xlim(x_values.min(),
            x_values.max())
    plt.ylim(y_values.min(),
            y_values.max())

    plt.xticks((np.arange(int(min(X[:, 0])-1), int(max(X[:, 0])+1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1])-1), int(max(X[:, 1])+1), 1.0)))

    plt.savefig('output/output.png')

def show_cifar(pixel_array):
    pixel_array = pixel_array.transpose(1, 2, 0)
    print pixel_array.shape
    plt.axis('off')
    plt.imshow(pixel_array, interpolation='catrom')
    plt.savefig('output/output.png')


def show_digit(pixel_array):
    pixels = np.array(pixel_array, dtype='uint8')

    plt.imshow(pixels, cmap='gray')
    plt.savefig('output/output.png')
