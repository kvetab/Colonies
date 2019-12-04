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

    inputData = CNNutils.load_photo('photos_used/'+photo, 98)

    graph = tf.get_default_graph()

    x = tf.placeholder(tf.float32, [None, 98, 98, 3])

    (f1, f2, f3) = filter_num
    if split_filters:
        (f1, f2, f3) = (int(f1/2), int(f2/2), int(f3))


    layer1, s1 = create_new_conv_layer(x, [2, 2], 1, outf_l, name='layer1', graph=graph)
    if split_filters:
        s1 = create_conv_layer_for_sum(x, 1, outf_s, name='s_layer1', graph=graph)

    layer2, s2 = create_new_conv_layer(layer1, [2, 2], 2, outf_l, name='layer2', graph=graph)
    if split_filters:
        s2 = create_conv_layer_for_sum(layer1, 2, outf_s, name='s_layer2', graph=graph)

    s3 = create_conv_layer_for_sum(layer2, 2, tf.nn.sigmoid, name='s_layer3', graph=graph)


    y_pred = s3
    for i, s in enumerate([s1, s2]):
        if which_sum[i] == 1:
            y_pred += s




    sum = tf.reduce_sum(s3)

    sum_ = sess.run(sum, feed_dict={x: inputData})
    print(sum_)


if __name__ == "__main__":
    LoadModel('model1573554663.131752', 'PICT9576.png', (10,20,30), True, (0, 1, 1), tf.nn.relu, CNN.sigmoid_ext)

