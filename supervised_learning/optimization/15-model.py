#!/usr/bin/env python3
"""Builds, trains, and saves a neural network model"""

import numpy as np
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data
create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999,
          epsilon=1e-8, decay_rate=1, batch_size=32,
          epochs=5, save_path='/tmp/model.ckpt'):
    """
    builds, trains, and saves a neural network model
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    nx = X_train.shape[1]
    classes = Y_train.shape[1]

    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    layer = x

    for i in range(len(layers)):
        if i == len(layers) - 1:
            init = tf.contrib.layers.variance_scaling_initializer(
                mode="FAN_AVG"
            )
            dense = tf.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_initializer=init
            )
            layer = dense(layer)
        else:
            layer = create_batch_norm_layer(
                layer,
                layers[i],
                activations[i]
            )

    y_pred = layer

    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    correct = tf.equal(
        tf.argmax(y_pred, 1),
        tf.argmax(y, 1)
    )

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

        m = X_train.shape[0]
        steps = (m + batch_size - 1) // batch_size

        for epoch in range(epochs + 1):

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

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_acc))

            if epoch == epochs:
                break

            X_shuff, Y_shuff = shuffle_data(
                X_train,
                Y_train
            )

            for step in range(steps):
                start = step * batch_size
                end = min(start + batch_size, m)

                X_batch = X_shuff[start:end]
                Y_batch = Y_shuff[start:end]

                sess.run(
                    train_op,
                    feed_dict={
                        x: X_batch,
                        y: Y_batch
                    }
                )

                step_num = step + 1

                if (step_num % 100 == 0 or
                        step_num == steps):

                    batch_cost, batch_acc = sess.run(
                        [loss, accuracy],
                        feed_dict={
                            x: X_batch,
                            y: Y_batch
                        }
                    )

                    print("\tStep {}:".format(step_num))
                    print("\t\tCost: {}".format(batch_cost))
                    print("\t\tAccuracy: {}".format(batch_acc))

        return saver.save(sess, save_path)
