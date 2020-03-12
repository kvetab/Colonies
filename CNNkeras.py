from tensorflow.keras.models import Model
from tensorflow.keras import losses, optimizers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
import tensorflow as tf
import numpy as np
import CNNutils
import os
from datetime import datetime
import csv


def create_model(learn_rate, epoch_num, batches, outf_layer, outf_sum, filter_num, split_filters, which_sum):
    input_shape = (98, 98, 3)
    inputs = Input(shape=input_shape, name='image_input')       #potrebuji toto?
    # filter number settings
    (f1, f2, f3) = filter_num
    # jen 3 filtry na sumaci
    if split_filters:
        (f1, f2, f3) = (int(f1 - 3), int(f2 - 3), int(f3))

    # normal layer
    convolution_1 = Conv2D(f1, kernel_size=(5, 5), strides=(1, 1), activation=outf_layer,
                           input_shape=input_shape, name='c_layer_1')(inputs)
    # zadne weights? zadny bias?
    s1 = tf.reduce_sum(convolution_1, axis=[1, 2, 3], name='c_layer_1_sum')
    pooling_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='p_layer_1')(convolution_1)
    if split_filters:
        # sum "layer"
        s1 = tf.reduce_sum(Conv2D(3, kernel_size=(5, 5), strides=(1, 1), activation=outf_sum,
                           input_shape=input_shape)(inputs), name='c_layer_1_sum')

    convolution_2 = Conv2D(f2, kernel_size=(5, 5), strides=(1, 1), activation=outf_layer,
                           input_shape=input_shape, name='c_layer_2')(pooling_1)
    s2 = tf.reduce_sum(convolution_2, axis=[1, 2, 3], name='c_layer_2_sum')
    pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='p_layer_2')(convolution_2)
    if split_filters:
        s2 = tf.reduce_sum(Conv2D(3, kernel_size=(5, 5), strides=(1, 1), activation=outf_sum,
                                  input_shape=input_shape)(pooling_1), name='c_layer_2_sum')

    convolution_3 = Conv2D(f3, kernel_size=(5, 5), strides=(1, 1), activation=outf_sum,
                           input_shape=input_shape, name='c_layer_3')(pooling_2)
    s3 = tf.reduce_sum(convolution_3, axis=[1, 2, 3], name='c_layer_3_sum')

    y_pred = s3
    for i, s in enumerate([s1, s2]):
        if which_sum[i] == 1:
            y_pred += s

    model = Model(inputs=inputs, outputs=s3)
    model.compile(loss=losses.MeanSquaredError, optimizer=optimizers.Adam,
                  metrics=['root_mean_squared_error', 'mean_absolute_error'])   # jak udelat metriku, kde se vyrusi + a - error?
    return model



model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])