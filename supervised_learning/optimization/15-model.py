#!/usr/bin/env python3
"""15-model"""

import tensorflow as tf
import numpy as np


def shuffle_data(X, Y):
    """shuffle data"""
    p = np.random.permutation(X.shape[0])
    return X[p], Y[p]


def create_batch_norm_layer(prev, n, activation):
    """batch normalization layer"""
    init = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )

    dense = tf.layers.Dense(
        units=n,
        activation=None,
        kernel_initializer=init
    )

    Z = dense(prev)

    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    mean, variance = tf.nn.moments(Z, axes=[0])

    Z_norm = tf.nn.batch_normalization(
        Z,
        mean,
        variance,
        beta,
        gamma,
        1e-8
    )

    if activation is None:
        return Z_norm

    return activation(Z_norm)


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999,
          epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """builds and trains model"""

    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    nx = X_train.shape[1]
    classes = Y_train.shape[1]

    x = tf.placeholder(tf.float32, shape=[None, nx])
    y = tf.placeholder(tf.float32, shape=[None, classes])

    layer = x

    for nodes, activation in zip(layers[:-1], activations[:-1]):
        layer = create_batch_norm_layer(
            layer,
            nodes,
            activation
        )

    init = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )

    output = tf.layers.Dense(
        units=layers[-1],
        activation=activations[-1],
        kernel_initializer=init
    )(layer)

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=output
    )

    prediction = tf.argmax(output, axis=1)
    labels = tf.argmax(y, axis=1)

    correct = tf.equal(prediction, labels)

    accuracy = tf.reduce_mean(
        tf.cast(correct, tf.float32)
    )

    global_step = tf.Variable(
        0,
        trainable=False
    )

    alpha_decay = tf.train.inverse_time_decay(
        alpha,
        global_step,
        1,
        decay_rate,
        staircase=True
    )

    train_op = tf.train.AdamOptimizer(
        learning_rate=alpha_decay,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    ).minimize(
        loss,
        global_step=global_step
    )

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_cost, train_acc = sess.run(
            [loss, accuracy],
            feed_dict={
                x: X_train,
                y: Y_train
            }
        )

        valid_cost, valid_acc = sess.run(
            [loss, accuracy],
            feed_dict={
                x: X_valid,
                y: Y_valid
            }
        )

        print("After 0 epochs:")
        print("\tTraining Cost: {}".format(train_cost))
        print("\tTraining Accuracy: {}".format(train_acc))
        print("\tValidation Cost: {}".format(valid_cost))
        print("\tValidation Accuracy: {}".format(valid_acc))

        for epoch in range(epochs):

            X_shuff, Y_shuff = shuffle_data(
                X_train,
                Y_train
            )

            m = X_train.shape[0]

            step = 0

            for i in range(0, m, batch_size):

                X_batch = X_shuff[i:i + batch_size]
                Y_batch = Y_shuff[i:i + batch_size]

                sess.run(
                    train_op,
                    feed_dict={
                        x: X_batch,
                        y: Y_batch
                    }
                )

                step += 1

                if step % 100 == 0:

                    batch_cost, batch_acc = sess.run(
                        [loss, accuracy],
                        feed_dict={
                            x: X_batch,
                            y: Y_batch
                        }
                    )

                    print("\tStep {}:".format(step))
                    print(
                        "\t\tCost: {}".format(
                            batch_cost
                        )
                    )
                    print(
                        "\t\tAccuracy: {}".format(
                            batch_acc
                        )
                    )

            train_cost, train_acc = sess.run(
                [loss, accuracy],
                feed_dict={
                    x: X_train,
                    y: Y_train
                }
            )

            valid_cost, valid_acc = sess.run(
                [loss, accuracy],
                feed_dict={
                    x: X_valid,
                    y: Y_valid
                }
            )

            print(
                "After {} epochs:".format(
                    epoch + 1
                )
            )
            print(
                "\tTraining Cost: {}".format(
                    train_cost
                )
            )
            print(
                "\tTraining Accuracy: {}".format(
                    train_acc
                )
            )
            print(
                "\tValidation Cost: {}".format(
                    valid_cost
                )
            )
            print(
                "\tValidation Accuracy: {}".format(
                    valid_acc
                )
            )

        return saver.save(
            sess,
            save_path
        )
