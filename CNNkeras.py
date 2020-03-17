from tensorflow.keras.models import Model
from tensorflow.keras import losses, optimizers, metrics
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
import tensorflow as tf
import CNNutils
import os
from datetime import datetime
import matplotlib.pyplot as plt
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
    model.compile(loss=losses.MeanSquaredError(),
                  optimizer=optimizers.Adam(learning_rate=learn_rate, name='Adam'),
                  metrics=[metrics.RootMeanSquaredError(), metrics.MeanAbsoluteError()])   # jak udelat metriku, kde se vyrusi + a - error?

    return model

class AccuracyHistory(tf.keras.callbacks.Callback):
    def __init__(self, num):
        super(AccuracyHistory, self).__init__()
        self.num = num

    def on_train_begin(self, logs={}):
        self.rmse = []
        self.mae = []


    def on_epoch_end(self, epoch, logs={}):
        self.rmse.append(logs.get('root_mean_squared_error'))
        self.mae.append(logs.get('mean_absolute_error'))
        with open("models/model" + str(self.num) + "/results.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow((epoch, logs.get('root_mean_squared_error'), logs.get('mean_absolute_error')))


def train_model(learn_rate, epoch_num, batches, outf_layer, outf_sum, filter_num, split_filters, which_sum, model, folder):
    X_train, X_test, y_train, y_test = CNNutils.load_input_data_as_np(folder+"/labels/labels.csv", folder)
    timer = datetime.now()
    num = timer.timestamp()
    now = datetime.now()
    os.mkdir("models/model" + str(num))
    save_path = ("models/model" + str(num))
    history = AccuracyHistory(num)

    with open(save_path + "/results.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow((num, learn_rate, epoch_num, batches, outf_layer.__name__, outf_sum.__name__, filter_num,
                         str(which_sum), split_filters))

    model.fit(X_train, y_train,
          batch_size=batches,
          epochs=epoch_num,
          verbose=2,
          validation_data=(X_test, y_test),
          callbacks=[history])

    model_json = model.to_json()
    with open(save_path + "/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(save_path + "/model.h5", overwrite=False, save_format="h5")
    model.save_weights(save_path + "/model.ckpt", overwrite=False, save_format="tf")
    print("Model saved in path: %s" % save_path)

    with open("results/results.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow((num, learn_rate, epoch_num, batches, outf_layer.__name__, outf_sum.__name__, filter_num,
                         str(which_sum), split_filters, history.rmse[-1]))

    plt.plot(range(1,epoch_num+1), history.rmse)
    plt.xlabel('Epochs')
    plt.ylabel('rmse')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(save_path+"/plot.png")

def sigmoid_shifted(x):
    x_ = (tf.nn.sigmoid(x) * 1.1) - 0.1
    return tf.nn.relu(x_)


def sigmoid_ext(x):
    x_ = (tf.nn.sigmoid(x) * 1.2) - 0.1
    return  x_




if __name__ == "__main__":
    learning_rate = 0.0001
    epochs = 10
    batch_size = 16
    outf_layer = tf.keras.activations.relu
    outf_sum = tf.keras.activations.relu
    filter_numbers = (6, 10, 16)
    split_filters = False
    what_to_sum = (0, 0, 1)
    #model = create_model(learning_rate, epochs, batch_size, outf_layer, outf_sum, filter_numbers, split_filters, what_to_sum)
    #train_model(learning_rate, epochs, batch_size, outf_layer, outf_sum, filter_numbers, split_filters, what_to_sum, model, "male")

