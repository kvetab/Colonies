import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import CNNutils
import math


def run_cnn():
    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    inputData = CNNutils.LoadInput()

    # Python optimisation variables
    learning_rate = 0.0001
    epochs = 10
    batch_size = 8

    # declare the training data placeholders
    # input x - probably sth like 98 x 98 pixels??
    x = tf.placeholder(tf.float32, [None, 28812])        # 9604 is just 98*98, not regarding 3 color dimensions... -> 28812
    # reshape the input data so that it is a 4D tensor.  The first value (-1) tells function to dynamically shape that
    # dimension based on the amount of data passed to it.  The two middle dimensions are set to the image size (i.e. 98
    # x 98).  The final dimension is 1 as there is only a single colour channel i.e. grayscale.  If this was RGB, this
    # dimension would be 3
    x_shaped = tf.reshape(x, [-1, 98, 98, 3])
    # now declare the output data placeholder - what is this going to be? Nothing, probably...
    y = tf.placeholder(tf.float32, [None, 10])
    #y = tf.placeholder(tf.float32)      #or can it be an integer??

    # create some convolutional layers
    layer1, s1 = create_new_conv_layer(x_shaped, 1, 15, [5, 5], [2, 2], 1, name='layer1')
    # input_data, num_input_channels, num_filters, filter_shape, pool_shape, stride, name
    # what happens to the 3 color dimensions? Is the output really 15 channels, or 3*15?
    layer2, s2 = create_new_conv_layer(layer1, 15, 30, [5, 5], [2, 2], 2, name='layer2')
    # and here?
    # what am I actually returning with the s1,2,3..?

    # another convolution added:
    layer3, s3 = create_new_conv_layer(layer2, 30, 40, [5, 5], [1,1], 2, name='layer3')  # no pooling

    # flatten the output ready for the fully connected output stage - after two layers of stride 2 pooling, we go
    # from 28 x 28, to 14 x 14 to 7 x 7 x,y co-ordinates, but with 64 output channels.  To create the fully connected,
    # "dense" layer, the new shape needs to be [-1, 7 x 7 x 64]
    flattened = tf.reshape(layer2, [-1, 4 * 4 * 40])
    # hopefully the result is really 4 x 4...

# I don't really need this, do I..
    """
    # fully connected layer
    # setup some weights and bias values for this layer, then activate with ReLU
    wd1 = tf.Variable(tf.truncated_normal([4 * 4 * 30, 1000], stddev=0.03), name='wd1')
    # how many nodes?
    bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
    dense_layer1 = tf.matmul(flattened, wd1) + bd1
    dense_layer1 = tf.nn.relu(dense_layer1)

    # another layer with softmax activations
    wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.03), name='wd2')
    bd2 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')
    dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
    y_ = tf.nn.softmax(dense_layer2)
    """

    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))
    #maybe I don't need this??
    #instead:
    y_pred = s1 + s2 + s3
    error = math.pow((y - y_pred), 2)

    # add an optimiser
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)
    #changed cross_entropy to error

    # define an accuracy assessment operation
    #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.equal(y, y_pred)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # setup recording variables
    # add a summary to store the accuracy
    """
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('C:\\Users\\Andy\\PycharmProjects')
    """

    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        total_batch = int(len(inputData.train.labels) / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = inputData.train.next_batch(batch_size=batch_size)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            test_acc = sess.run(accuracy, feed_dict={x: inputData.test.images, y: inputData.test.labels})
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(test_acc))
            #summary = sess.run(merged, feed_dict={x: inputData.test.images, y: inputData.test.labels})
            #writer.add_summary(summary, epoch)

        print("\nTraining complete!")
        #writer.add_graph(sess.graph)
        print(sess.run(accuracy, feed_dict={x: inputData.test.images, y: inputData.test.labels}))
# changed all uses of mnist to inputData - will have to make same functionalitu for loading dats


def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, stride, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    # initialise weights and bias for the filter
    #decreased SD
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.01), name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, stride, stride, 1], padding='VALID')
    #changed padding - was that right??

    # add the bias
    out_layer += bias

    sum_ = tf.math_ops.reduce_sum(out_layer)
    # what happens here to the dynamically changed dimension??
    # !!!

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    # ksize is the argument which defines the size of the max pooling window (i.e. the area over which the maximum is
    # calculated).  It must be 4D to match the convolution - in this case, for each image we want to use a 2 x 2 area
    # applied to each channel
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    # strides defines how the max pooling area moves through the image - a stride of 2 in the x direction will lead to
    # max pooling areas starting at x=0, x=2, x=4 etc. through your image.  If the stride is 1, we will get max pooling
    # overlapping previous max pooling areas (and no reduction in the number of parameters).  In this case, we want
    # to do strides of 2 in the x and y directions.
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
    #here I left the padding, not sure why..

    return out_layer, sum_

if __name__ == "__main__":
    run_cnn()