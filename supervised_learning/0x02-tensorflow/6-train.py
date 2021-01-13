#!/usr/bin/env python3
""" doc """
import tensorflow as tf


calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train,
          X_valid, Y_valid,
          layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """ doc """
    nx = X_train.shape[1]
    ny = Y_train.shape[1]
    # Placeholders
    x, y = create_placeholders(nx, ny)
    # forward propagation
    y_pred = forward_prop(x, layer_sizes, activations)
    # softmax cross entropy
    loss = calculate_loss(y, y_pred)
    # accuracy
    acc = calculate_accuracy(y, y_pred)
    # train operation
    train_op = create_train_op(loss, alpha)
    # Starting the Tensorflow Session 
    sess = tf.Session()
    # Initializing the Variables 
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    # iterations
    for i in range(iterations+1):
        # Calculating costs && accuracies on current iteration 
        cost_train = loss.eval({x : X_train, y : Y_train}, sess)
        accuracy_train = acc.eval({x : X_train, y : Y_train}, sess)
        cost_valid = loss.eval({x : X_valid, y : Y_valid}, sess)
        accuracy_valid = acc.eval({x : X_valid, y : Y_valid}, sess)
        # Displaying training result on current iteration
        if (i % 100 == 0):
            print("After {} iterations:".format(i)
                  +"\n\tTraining Cost: {}".format(cost_train)
                  + "\n\tTraining Accuracy: {}".format(accuracy_train)
                  + "\n\tValidation Cost: {}".format(cost_valid))
        if (i != iterations):
            # Training data
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})
            sess.run(train_op, feed_dict={x: X_valid, y: Y_valid})
    # Save Training session
    trainSaver = tf.train.Saver()
    return trainSaver.save(sess, save_path)
