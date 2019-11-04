import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

import CNNutils
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import csv





def run_cnn(learn_rate, epoch_num, batches, outf_layer, outf_sum, filter_num, split_filters, which_sum):
    inputData = CNNutils.LoadInputIMG("/mnt/0/labels/labels.csv")

    # Python optimisation variables
    #learning_rate = 0.0001
    #epochs = 20
    #batch_size = 16
    learning_rate = learn_rate
    epochs = epoch_num
    batch_size = batches

    # filter number settings
    (f1, f2, f3) = filter_num
    if split_filters:
        (f1, f2, f3) = (int(f1/2), int(f2/2), int(f3))

    # declare the training data placeholders
    x = tf.placeholder(tf.float32, [None, 98, 98, 3])        # 98*98 plus 3 color dimensions... -> 28812
    # output data placeholder
    y = tf.placeholder(tf.float32, [None, ])      #or can it be an integer??

    # create some convolutional layers
    layer1, s1 = create_new_conv_layer(x, 3, f1, [5, 5], [2, 2], 1, outf_layer, name='layer1')
    # input_data, num_input_channels, num_filters, filter_shape, pool_shape, stride, out_fction, name
    # input_channels is 3 because of RGB
    if split_filters:
        s1 = create_conv_layer_for_sum(x, 3, f1, [5, 5], 1, outf_sum, name='s_layer1')

    layer2, s2 = create_new_conv_layer(layer1, f1, f2, [5, 5], [2, 2], 2, outf_layer, name='layer2')
    if split_filters:
        s2 = create_conv_layer_for_sum(layer1, f1, f2, [5, 5], 2, outf_sum, name='s_layer2')

    # another convolution added:
    s3 = create_conv_layer_for_sum(layer2, f2, f3, [5, 5], 2, outf_sum, name='s_layer3')



    # prediction:
    y_pred = s3
    for i, s in enumerate([s1, s2]):
        if which_sum[i] == 1:
            y_pred += s
    #y_pred = tf.multiply(s1, which_sum[0]) + tf.multiply(s2, which_sum[1]) + tf.multiply(s3, which_sum[2])
    #y_pred = s1 + s2 + s3
    # option to add different sums (all/some layers)
    error = tf.pow((y - y_pred), 2)
    err_mean = tf.sqrt(tf.reduce_mean(error))
    abs_error = y - y_pred

    # add an optimiser
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)
    #changed cross_entropy to error

    # define an accuracy assessment operation
    #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #correct_prediction = tf.equal(y, y_pred)
    rmse = tf.pow((y - y_pred), 2)      # same as error
    # pri krokovani se mi zda, ze to ma jinou hodnotu nez error, ackoli se to pocita stejne
    accuracy = tf.sqrt(tf.reduce_mean(rmse))    # same as err_mean
    abs_err_mean = tf.reduce_mean(abs_error)

    # setup the initialisation operator
    init_op = tf.global_variables_initializer()


    with tf.Session() as sess:
        timer = datetime.now()
        num = timer.timestamp()
        now = datetime.now()
        os.mkdir("/mnt/0/models/model" + str(num))

        # initialise the variables
        sess.run(init_op)
        total_batch = int(len(inputData.train.labels) / batch_size)

        for epoch in range(epochs):
            avg_cost = 0
            avg_abs_cost = 0        #added second error for gaining inf about distribution of error
            print("total_batch",total_batch)
            for i in range(total_batch):
                batch_x, batch_y = inputData.train.next_batch(batch_size=batch_size)
                _, c, err = sess.run([optimiser, err_mean, abs_err_mean], feed_dict={x: batch_x, y: batch_y})

                avg_cost += c / total_batch
                avg_abs_cost += err / total_batch
            test_acc, abs_test_acc, y_solv, y_pred_solv = sess.run([accuracy, abs_err_mean, y, y_pred], feed_dict={x: inputData.test.images, y: inputData.test.labels})
            #print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), " test accuracy:", "{:.3f}".format(test_acc))
            print("Epoch: ", str((epoch + 1)), "cost = ", str( avg_cost), "abs cost = ", str(avg_abs_cost), "test accuracy: ", str(test_acc), "abs accuracy = ", str(abs_test_acc))
            print(list(zip(y_solv,y_pred_solv))[0:20])

            frequency = 20
            if epoch % frequency == 0:
                new_now = datetime.now()
                length = new_now - now
                now = new_now
                with open("/mnt/0/models/model"+str(num)+"/results.csv", 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow((epoch, avg_cost, avg_abs_cost, test_acc, abs_test_acc, length.microseconds/frequency))
                #LP: store trained filters into an image
                #getActivations(sess, layer1, inputData.test.images[4:8], x, epoch, "layer1")
                #getActivations(sess, layer2, inputData.test.images[4:8], x, epoch, "layer2")

            #print(y_solv)
            #print(y_pred_solv)
            # ZDE KONEC KROKOVANI
            #summary = sess.run(merged, feed_dict={x: inputData.test.images, y: inputData.test.labels})
            #writer.add_summary(summary, epoch)


# SAVING NETWORK
        print("\nTraining complete!")
        saver = tf.train.Saver()
        save_path = saver.save(sess, "/mnt/0/models/model"+str(num)+"/model.ckpt")
        print("Model saved in path: %s" % save_path)
        print(learn_rate, epoch_num, batches, outf_layer.__name__, outf_sum.__name__, filter_num, split_filters)

        #writer.add_graph(sess.graph)
        acc = sess.run(accuracy, feed_dict={x: inputData.test.images, y: inputData.test.labels})
        print(acc)

        resf = "results.csv"
        with open("/mnt/0/results/" + resf, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow((num, learn_rate, epoch_num, batches, outf_layer.__name__, outf_sum.__name__, filter_num, split_filters, acc))


def getActivations(sess, layer, stimuli, x, iter, labelname):
    units = sess.run(layer, feed_dict={x: stimuli})
    plotNNFilter(units, stimuli, iter, labelname)


def plotNNFilter(units, stimuli, iter, labelname):
    filters = units.shape[3]
    images = units.shape[0]
    #print(units.shape)
    n_columns = images
    n_rows = filters+1
    fig, axes = plt.subplots(n_columns, n_rows, figsize=(n_rows*2, n_columns*2))

    for i in range(filters):
        for j in range(images):
            #print(units[j, :, :, i].shape)
            #print(stimuli[j].shape)
            ax = sns.heatmap(units[j, :, :, i],
                               cmap=matplotlib.cm.Greens,
                               alpha=1.0,  # whole heatmap is translucent
                               annot=False,
                               zorder=2,
                               yticklabels=False,
                               xticklabels=False,
                               ax=axes[j, i+1],
                               cbar = False,
                               linewidths=0.0
                               )

            # heatmap uses pcolormesh instead of imshow, so we can't pass through
            # extent as a kwarg, so we can't mmatch the heatmap to the map. Instead,
            # match the map to the heatmap:
            ax.set_ylabel('')
            ax.set_xlabel('')
            """ax.imshow(stimuli[j],
                        aspect=ax.get_aspect(),
                        extent=ax.get_xlim() + ax.get_ylim(),
                        interpolation="bilinear",
                        zorder=1)  # put the map under the heatmap"""

    for j in range(images):
        axes[j, 0].imshow(stimuli[j],
                          aspect=ax.get_aspect(),
                          extent=ax.get_xlim() + ax.get_ylim(),
                          interpolation="bilinear",
                          #ax = axes[i+1, j],
                          zorder=1)
        axes[j, 0].set_ylabel('')
        axes[j, 0].set_xlabel('')

    fig.savefig("filters/"+str(iter)+"_"+str(labelname)+".png", dpi=150)
    print("Labels saved")


# output function variants
def sigmoid_shifted(x):
    x_ = (tf.nn.sigmoid(x) * 1.1) - 0.1
    return tf.nn.relu(x_)


def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, stride, out_fction, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    # initialise weights and bias for the filter
    #decreased SD
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.001), name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, stride, stride, 1], padding='VALID')
    # add the bias
    out_layer += bias

    #LP: pred sectenim vystupu jsem je nechal prolozit pomoci sigmoidy - tedy da se to chapat jako pravdepodobnost,
    # ze dany filter zachytil label. Zatim nevim nakolik je tahle uprava rozumna, ale vypada to, ze to vcelku funguje
    #sigmoid_layer = tf.nn.sigmoid(out_layer)
    # ale nepouziva se pro vystup, jen pro sumu..?

    # apply a ReLU non-linear activation

    out_layer = out_fction(out_layer)


    #LP: presunul jsem po aplikaci RELU
    # - tady je potreba zachovat prvni dimenzi, jinak se to snazi odhadnout velikost cele batche
    sum_ = tf.reduce_sum(out_layer, axis=[1, 2, 3])

    # now perform max pooling - ksize is window size
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
    #here I left the padding, not sure why..

    return out_layer, sum_


def create_conv_layer_for_sum(input_data, num_input_channels, num_filters, filter_shape, stride, out_fction, name):

    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.001), name=name + '_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name + '_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, stride, stride, 1], padding='VALID')
    # add the bias
    out_layer += bias

    transformed = out_fction(out_layer)
    sum_ = tf.reduce_sum(transformed, axis=[1, 2, 3])

    return sum_

if __name__ == "__main__":
    print("bezi kod v CNN.py")
    run_cnn(0.0001, 20, 16, tf.nn.relu, tf.nn.sigmoid, (1, 2, 3), False, (0, 0, 1))
    #run_cnn(learn_rate, epoch_num, batches, outf_layer, outf_sum, filter_num, split_filters, which_sum)

