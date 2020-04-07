from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras import losses, optimizers, metrics
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, AveragePooling2D, concatenate
import tensorflow as tf
from bin import CNNutils
import os
from datetime import datetime
import matplotlib.pyplot as plt
import csv


def create_model(learn_rate, epoch_num, batches, outf_layer, outf_sum, filter_num, split_filters, which_sum, fc):
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

    if fc:
        flat = Flatten()(convolution_3)
        s3 = Dense(1, activation=outf_sum)(flat)  # pouzit outf_layer nebo sum? Nastavovat i neco dalsiho??
    else:
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


def create_model_mean_pooling(learn_rate, epoch_num, batches, outf_layer, outf_sum, filter_num, split_filters, which_sum, fc):
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
    s1 = tf.reduce_sum(convolution_1, axis=[1, 2, 3], name='c_layer_1_sum')
    a1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='p_layer_1')(convolution_1)
    if split_filters:
        # sum "layer"
        s1 = tf.reduce_sum(Conv2D(3, kernel_size=(5, 5), strides=(1, 1), activation=outf_sum,
                           input_shape=input_shape)(inputs), name='c_layer_1_sum')

    a2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(a1)

    a3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(a2)

    K = Conv2D(f2, kernel_size=(5, 5), strides=(1, 1), activation=outf_layer,
               name='c_layer_2')
    # tady nevim, jestli nebude problem s tim input shape, protoze ten se teoreticky meni - uvidis jestli to pujde, pripadne jestli je mozne input size vynechat...

    v1 = K(a1)
    v2 = K(a2)
    v3 = K(a3)

    # scitas v1, v2, v3
    flat1 = Flatten()(v1)
    flat2 = Flatten()(v2)
    flat3 = Flatten()(v3)
    merged = concatenate([flat1, flat2, flat3])

    if fc:
        # tady potrebujes udelat konkatenaci v1, v2 a v3 (neco jako merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1))
        # aktivace by urcite nemela být sigmoida (ta dává výsledek mezi 0-1). Možná tady by bylo ideální použít klasické ReLU, protože mín jak nula kolonií to mít nebude...
        s3 = Dense(1, activation=tf.keras.activations.relu)(merged)
    else:
        # opet mužeš použít konkatenaci v1, v2, v3 a secíst
        summed = tf.reduce_sum(merged, axis=[1])

        s3 = summed



    model = Model(inputs=inputs, outputs=s3)
    model.compile(loss=losses.MeanSquaredError(),
                  optimizer=optimizers.Adam(learning_rate=learn_rate, name='Adam'),
                  metrics=[metrics.RootMeanSquaredError(), metrics.MeanAbsoluteError()])

    return model


class AccuracyHistory(tf.keras.callbacks.Callback):
    def __init__(self, num):
        super(AccuracyHistory, self).__init__()
        self.num = num

    def on_train_begin(self, logs={}):
        self.rmse = []
        self.mae = []
        self.test_acc = 0


    def on_epoch_end(self, epoch, logs={}):
        self.rmse.append(logs.get('root_mean_squared_error'))
        self.mae.append(logs.get('mean_absolute_error'))
        with open("models/pokusy/model" + str(self.num) + "/results.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow((epoch, logs.get('root_mean_squared_error'), logs.get('mean_absolute_error')))



def train_model(learn_rate, epoch_num, batches, outf_layer, outf_sum, filter_num, split_filters, which_sum, model, folder, fc, mean):
    X_train, X_test, y_train, y_test = CNNutils.load_input_data_as_np(folder + "/labels/labels.csv", folder)
    timer = datetime.now()
    num = timer.timestamp()
    now = datetime.now()
    os.mkdir("models/pokusy/model" + str(num))
    save_path = ("models/pokusy/model" + str(num))
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

    results = model.evaluate(X_test, y_test, batch_size=16)
    print('test loss, test acc:', results)

    model_json = model.to_json()
    with open(save_path + "/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(save_path + "/model.h5", overwrite=False, save_format="h5")
    model.save_weights(save_path + "/model.ckpt", overwrite=False, save_format="tf")
    print("Model saved in path: %s" % save_path)

    fc_string = "FC" if fc else "no_fc"
    mean_string = "mean" if mean else "max"
    with open("../results/pokusy/results.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow((num, learn_rate, epoch_num, batches, outf_layer.__name__, outf_sum.__name__, filter_num,
                         str(which_sum), split_filters, fc_string, mean_string, results[0], results[1], results[2]))

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

def test_model(model_dir, learn_rate):
    json_file = open(model_dir + "/model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_number = model_dir.replace("models/model", "")
    model = model_from_json(loaded_model_json)
    model.load_weights(model_dir + "/model.h5")
    model.compile(loss=losses.MeanSquaredError(),
                  optimizer=optimizers.Adam(learning_rate=learn_rate, name='Adam'),
                  metrics=[metrics.RootMeanSquaredError(), metrics.MeanAbsoluteError()])

    images, labels = CNNutils.load_test_data()

    results = model.evaluate(images, labels, batch_size=16)
    print('test loss, test acc:', results)

    with open("../results/test_results.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow((model_number, results[0], results[1], results[2]))


def pipeline(learn_rate, epoch_num, batches, outf_layer, outf_sum, filter_num, split_filters, which_sum, folder, fc, mean):
    if mean:
        model = create_model_mean_pooling(learn_rate, epoch_num, batches, outf_layer, outf_sum, filter_num, split_filters, which_sum, fc)
    else:
        model = create_model(learn_rate, epoch_num, batches, outf_layer, outf_sum, filter_num, split_filters, which_sum, fc)
    train_model(learn_rate, epoch_num, batches, outf_layer, outf_sum, filter_num, split_filters, which_sum, model, folder, fc, mean)




if __name__ == "__main__":
    learning_rate = 0.0001
    epochs = 10
    batch_size = 16
    outf_layer = tf.keras.activations.relu
    outf_sum = tf.keras.activations.relu
    filter_numbers = (6, 10, 16)
    split_filters = False
    what_to_sum = (0, 0, 1)
    #pipeline(learning_rate, epochs, batch_size, outf_layer, outf_sum, filter_numbers, split_filters, what_to_sum, "male", False, True)
    #model = create_model(learning_rate, epochs, batch_size, outf_layer, outf_sum, filter_numbers, split_filters, what_to_sum)
    #train_model(learning_rate, epochs, batch_size, outf_layer, outf_sum, filter_numbers, split_filters, what_to_sum, model, "male")
    test_model("models/model" + "1585480554.82895", learning_rate)
    test_model("models/model" + "1585482634.665865", learning_rate)
    test_model("models/model" + "1585482634.665865", learning_rate)
    test_model("models/model" + "1585734946.984284", learning_rate)
    test_model("models/model" + "1585737032.612351", learning_rate)












