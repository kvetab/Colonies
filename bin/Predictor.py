import tensorflow as tf
import CNNutils
import csv
from tensorflow.keras import backend
from tensorflow.keras.models import model_from_json
import os
from CNNkeras import test_model

from CNNkeras import sigmoid_ext, sigmoid_shifted

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


        rmse = tf.pow((y - self.y_pred), 2)  # same as error

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

    inputData = CNNutils.load_photo('photos_used/' + photo, 98)
    input_test = CNNutils.LoadInputIMG("male/labels/labels.csv")

    x = graph.get_tensor_by_name('x:0')
    y = graph.get_tensor_by_name('y:0')


    s3 = graph.get_tensor_by_name('s_layer3_output:0')

    y_pred = s3
    sum_ = tf.reduce_sum(y_pred)

    pred, sum2 = sess.run([y_pred, sum_], feed_dict={x: inputData})
    print(pred)
    print(sum2)


class PredictorKeras:
    def __init__(self, model_dir):
        json_file = open(model_dir + "/model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model_dir = model_dir
        self.model_number = model_dir.split("/")[-1]
        self.model_number = self.model_number.split("\\")[-1]
        self.model = model_from_json(loaded_model_json, custom_objects={'sigmoid_ext': sigmoid_ext, 'sigmoid_shifted': sigmoid_shifted})
        self.model.load_weights(model_dir + "/model.h5")



    def predict(self, photo, verbose=1):
        inputData = CNNutils.load_photo( photo, 98)

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

    def evaluation(self, data_folder):
        input_data, labels = CNNutils.load_test_data(data_folder + "/labels/labels.csv", data_folder + "/test_crops/")
        prediction = self.model.predict(input_data)
        prediction = tf.keras.backend.eval(prediction)
        prediction = prediction.flatten()
        differences = prediction - labels
        with open(os.path.join(self.model_dir, "eval.txt"), "a") as outfile:
            for i in range(len(labels)):
                outstr = f"i: {i}; label: {labels[i]}; pred: {prediction[i]}, error: {differences[i]} \n"
                outfile.write(outstr)
        with open(os.path.join(self.model_dir, "diffs.txt"), "a") as outfile:
            for diff in differences:
                outfile.write(str(diff) + "\n")

    def test_on_image(self, filename):
        count = get_real_count(filename)
        prediction, positive = self.predict(filename, verbose=0)
        print("Real count is {}".format(count))
        print("Predicted count is {} or {}".format(prediction, positive))
        diff = abs(count - prediction)
        print("Difference is {}".format(diff))
        perc_error = diff/count * 100
        print("Error percentage is {}".format(perc_error))
        print()
        out_file = "predictions/predictions_" + self.model_number + ".txt"
        with open(out_file, 'a') as f:
            f.writelines(["Count: {}; Prediction: {}; Difference: {}; Percentage of error: {} \n".format(count, prediction, diff, perc_error)])


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
    learning_rate = 0.0001
    photo_list = ['PICT9620.png', 'PICT9575.png', 'PICT9563.png', 'PICT9567.png', 'PICT9612.png',
                  'PICT20190923_150344.png', 'PICT20190923_151541.png']
    for dir in os.listdir("models/dodatek"):
        predictor = PredictorKeras("models/dodatek/" + dir)
        predictor.evaluation("new_photos")

