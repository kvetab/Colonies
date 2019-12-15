import tensorflow as tf
import CNNutils
import CNN


def create_new_conv_layer(input_data, pool_shape, stride, out_fction, name, graph):
    weights = graph.get_tensor_by_name(name+'_W:0')
    bias = graph.get_tensor_by_name(name+'_b:0')
    out_layer = tf.nn.conv2d(input_data, weights, [1, stride, stride, 1], padding='VALID')
    out_layer += bias
    out_layer = out_fction(out_layer)
    sum_ = tf.reduce_sum(out_layer, axis=[1, 2, 3])
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

    return out_layer, sum_


def create_conv_layer_for_sum(input_data, stride, out_fction, name, graph):
    weights = graph.get_tensor_by_name(name + '_W:0')
    bias = graph.get_tensor_by_name(name + '_b:0')
    out_layer = tf.nn.conv2d(input_data, weights, [1, stride, stride, 1], padding='VALID')
    out_layer += bias
    transformed = out_fction(out_layer)
    sum_ = tf.reduce_sum(transformed, axis=[1, 2, 3])

    return sum_


def LoadModel(model, photo, filter_num, split_filters, which_sum, outf_l, outf_s):
    sess = tf.Session()

    saver = tf.train.import_meta_graph('models/' + model + '/model.ckpt.meta')
    saver.restore(sess, 'models/' + model + '/model.ckpt')

    seznam = [n.name for n in tf.get_default_graph().as_graph_def().node]
    #for i in seznam:
        #print(i)


    inputData = CNNutils.load_photo('photos_used/'+photo, 98)
    input_test = CNNutils.LoadInputIMG("labels/labels.csv")

    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name('x:0')
    y = graph.get_tensor_by_name('y:0')


    s3 = graph.get_tensor_by_name('s_layer3_output:0')





    sum = tf.reduce_sum(s3)
    y_pred = sum
    """
    abs_error = y - y_pred
    rmse = tf.pow((y - y_pred), 2)  # same as error
    # pri krokovani se mi zda, ze to ma jinou hodnotu nez error, ackoli se to pocita stejne
    accuracy = tf.sqrt(tf.reduce_mean(rmse))  # same as err_mean
    abs_err_mean = tf.reduce_mean(abs_error)

    acc = sess.run(accuracy, feed_dict={x: input_test.test.images, y: input_test.test.labels})
    abs_test_acc = sess.run(abs_err_mean, feed_dict={x: input_test.test.images, y: input_test.test.labels})
    print(acc, abs_test_acc)
    """

    sum_, s_ = sess.run([sum, s3], feed_dict={x: inputData})
    print(sum_)
    print(s_)


if __name__ == "__main__":
    LoadModel('model1576328186.405176', 'PICT9576.png', (10,20,30), True, (0, 0, 1), tf.nn.relu, CNN.sigmoid_ext)

