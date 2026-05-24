#!/usr/bin/env python3
"""Evaluate MNIST model predictions and display sample results."""

evaluate = __import__('7-evaluate').evaluate


def one_hot(y, classes):
    """convert an array to a one-hot matrix"""
    import numpy as np
    one_hot_matrix = np.zeros((y.shape[0], classes))
    one_hot_matrix[np.arange(y.shape[0]), y] = 1
    return one_hot_matrix


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    lib = np.load('../data/MNIST.npz')
    X_test_3D = lib['X_test']
    Y_test = lib['Y_test']
    X_test = X_test_3D.reshape((X_test_3D.shape[0], -1))
    y_test_oh = one_hot(Y_test, 10)

    Y_pred_oh, accuracy, cost = evaluate(X_test, y_test_oh, './model.ckpt')
    print("Test Accuracy:", accuracy)
    print("Test Cost:", cost)

    Y_pred = np.argmax(Y_pred_oh, axis=1)

    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_test_3D[i])
        plt.title(str(Y_test[i]) + ' : ' + str(Y_pred[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
