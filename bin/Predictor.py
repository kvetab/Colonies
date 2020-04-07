import tensorflow as tf
from bin import CNNutils
import csv
from tensorflow.keras import backend
from tensorflow.keras.models import model_from_json
import os


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


class Predictor:
    def __init__(self, model):
        self.sess = tf.Session()
        self.graph = self.sess.graph
        saver = tf.train.import_meta_graph('models/' + model + '/model.ckpt.meta')
        saver.restore(self.sess, 'models/' + model + '/model.ckpt')
        self.x = self.graph.get_tensor_by_name('x:0')
        y = self.graph.get_tensor_by_name('y:0')

        s3 = self.graph.get_tensor_by_name('s_layer3_output:0')

        self.y_pred = s3
        self.sum_ = tf.reduce_sum(self.y_pred)
        zeroes = tf.zeros_like(self.y_pred)
        self.sum_pos = tf.reduce_sum(tf.maximum(zeroes, self.y_pred))

        abs_error = y - self.y_pred

        rmse = tf.pow((y - self.y_pred), 2)  # same as error
        accuracy = tf.sqrt(tf.reduce_mean(rmse))  # same as err_mean
        abs_err_mean = tf.reduce_mean(abs_error)

    def predict_photo(self, filename):
        tiles = CNNutils.load_photo('photos_used/' + filename, 98)
        print("Predicting photo {}".format(filename))
        pred, sum2, sum_pos2 = self.sess.run([self.y_pred, self.sum_, self.sum_pos], feed_dict={self.x: tiles})
        #print(pred)
        print(sum2)
        print(sum_pos2)


def LoadModel(model, photo):
    sess = tf.Session()
    graph = sess.graph
    saver = tf.train.import_meta_graph('models/' + model + '/model.ckpt.meta')
    saver.restore(sess, 'models/' + model + '/model.ckpt')

    seznam = [n.name for n in tf.get_default_graph().as_graph_def().node]
    #for i in seznam:
        #print(i)

    inputData = CNNutils.load_photo('photos_used/' + photo, 98)
    input_test = CNNutils.LoadInputIMG("male/labels/labels.csv")

    x = graph.get_tensor_by_name('x:0')
    y = graph.get_tensor_by_name('y:0')


    s3 = graph.get_tensor_by_name('s_layer3_output:0')

    y_pred = s3
    sum_ = tf.reduce_sum(y_pred)

    abs_error = y - y_pred

    rmse = tf.pow((y - y_pred), 2)  # same as error
    accuracy = tf.sqrt(tf.reduce_mean(rmse))  # same as err_mean
    abs_err_mean = tf.reduce_mean(abs_error)

    #acc, abs_test_acc, predsum, pred = sess.run([accuracy, abs_err_mean, y_pred, s3], feed_dict={x: input_test.test.images, y: input_test.test.labels})
    #print(acc, abs_test_acc)
    pred, sum2 = sess.run([y_pred, sum_], feed_dict={x: inputData})
    print(pred)
    print(sum2)


class PredictorKeras:
    def __init__(self, model_dir):
        json_file = open(model_dir + "/model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model_number = model_dir.replace("models/model", "")
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(model_dir + "/model.h5")
        graph = tf.compat.v1.get_default_graph()        # Toto jsem opsala od Misi, ale nevim, k cemu to pouzivala a k cemu bych to mela pouzit ja.



    def predict(self, photo, verbose=1):
        # inputData = CNNutils.load_photo('photos_used/'+photo, 98)
        inputData = CNNutils.load_photo(photo, 98)

        s3 = self.model.predict(inputData)
        y_pred = s3
        zeroes = tf.zeros_like(y_pred)
        sum_ = tf.reduce_sum(y_pred)
        sum_positive = tf.reduce_sum(tf.maximum(zeroes, y_pred))

        if verbose:
            print("Predictions for tiles: ", s3)
            print("Sum for image: ", sum_)
            print("Sum of positive numbers: ", sum_positive)

        return backend.get_value(x=sum_), backend.get_value(x=sum_positive)

    def test_on_image(self, filename):
        count = get_real_count(filename)
        prediction, positive = self.predict(filename, verbose=0)
        print("Real count is {}".format(count))
        print("Predicted count is {} or {}".format(prediction, positive))
        diff = abs(count - prediction)
        print("Difference is {}".format(diff))
        print()
        out_file = "predictions/predictions_" + self.model_number + ".txt"
        with open(out_file, 'a') as f:
            f.writelines(["Count: {}; Prediction: {}; Difference: {} \n".format(count, prediction, diff)])


def get_real_count(photo):
    COORDS_DCT = "coords/"
    coords_file = photo.replace("PICT","coords").replace("png","csv")
    count = 0
    try:
        with open(COORDS_DCT + coords_file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                if line: count += 1
    except:
        return "undefined"
    print("Image {} contains {} colonies.".format(photo, count))
    return count




if __name__ == "__main__":
    #LoadModel('model1580577367.772957', 'PICT9620.png')
    #LoadModel('model1580554550.725145', 'PICT9575.png', (10, 20, 30), True, (0, 0, 1), tf.nn.relu, CNN.sigmoid_ext)
    photo_list = os.listdir("../new_photos")
    for model in os.listdir("../models"):
        try:
            predictor = PredictorKeras("models/" + model)
            for photo in photo_list:
                predictor.test_on_image(photo)
        except:
            pass

"""
    predictor = PredictorKeras("models/model" + "1585480554.82895")
    #photo_list = ['PICT9620.png', 'PICT9575.png', 'PICT9563.png', 'PICT9567.png', 'PICT9612.png',
    #              'PICT20190923_150344.png', 'PICT20190923_151541.png']
    for photo in photo_list:
        predictor.test_on_image(photo)

    predictor = PredictorKeras("models/model" + "1585482634.665865")
    for photo in photo_list:
        predictor.test_on_image(photo)

    predictor = PredictorKeras("models/model" + "1585386250.530946")
    for photo in photo_list:
        predictor.test_on_image(photo)
"""
