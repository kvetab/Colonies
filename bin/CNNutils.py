import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from matplotlib import pyplot

# returns image as a numpy array
def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    data = data / 255
    # scaling values to (0, 1)
    return data

def load_input_data_as_np(label_file, folder):
    labels = pd.read_csv(label_file, header=None)
    labels.columns = ["img", "label"]
    images = []
    for i in labels.img:
        numpy_img = load_image(folder+"/crops/" + i)
        images.append(numpy_img[:, :, 0:3])
        # print(numpy_img.shape)
    np_images = np.stack(images)
    np_labels = labels.label.to_numpy(copy=True)
    X_train, X_test, y_train, y_test = train_test_split(np_images, np_labels, test_size = 0.1, random_state = 42)
    return X_train, X_test, y_train, y_test



def load_test_data(file, folder):
    labels = pd.read_csv(file, header=None)
    labels.columns = ["img", "label"]
    images = []
    for i in labels.img:
        numpy_img = load_image(folder + i)
        images.append(numpy_img[:, :, 0:3])
        # print(numpy_img.shape)
    np_images = np.stack(images)
    np_labels = labels.label.to_numpy(copy=True)
    return np_images, np_labels


# takes all image filenames from label file and loads the images
# splits images and labels into test and train sets
# returns input_data structure, which has next_batch function defined
def LoadInputIMG(file_labels):
    labels = pd.read_csv(file_labels, header=None)
    labels.columns = ["img","label"]

    images = []
    for i in labels.img:
        numpy_img = load_image("male/crops/"+i)
        images.append(numpy_img[:,:,0:3])   # LP:hack to get rid of alpha in case of RGBA
    np_images = np.stack(images)

    np_labels = labels.label.to_numpy(copy=True)
    X_train, X_test, y_train, y_test = train_test_split(np_images, np_labels, test_size = 0.1, random_state = 42)

    class set:
        def __init__(self, x, y):
            assert x.shape[0] == y.shape[0], (
                    'images.shape: %s labels.shape: %s' % (x.shape, y.shape))
            self._num_examples = x.shape[0]
            self.images = x
            self.labels = y
            self._epochs_completed = 0
            self._index_in_epoch = 0

        def next_batch(self, batch_size):
        # Return the next `batch_size` examples from this data set.
        # copied from https://github.com/tensorflow/tensorflow/blob/7c36309c37b04843030664cdc64aca2bb7d6ecaa/tensorflow/contrib/learn/python/learn/datasets/mnist.py#L160
            start = self._index_in_epoch
            self._index_in_epoch += batch_size
            if self._index_in_epoch > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                # Shuffle the data
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self.images = self.images[perm]
                self.labels = self.labels[perm]
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
                assert batch_size <= self._num_examples
            end = self._index_in_epoch
            return self.images[start:end], self.labels[start:end]

    class input_data:
        def __init__(self, xtrain, ytrain, xtest, ytest):
            self.train = set(xtrain, ytrain)
            self.test = set(xtest, ytest)


    data = input_data(X_train, y_train, X_test, y_test)
    return data


# loads picture and slices it into size x size tiles, returns stacked as a np array
def load_photo(filename, size):
    img = Image.open(filename)
    img = img.resize((980, 980), Image.ANTIALIAS)
    im = np.asarray(img, dtype="int32")
    im = im / 255
    # scaling values to (0, 1)
    tiles = [im[x:x + size, y:y + size] for x in range(0, im.shape[0], size) for y in range(0, im.shape[1], size)]
    #print(tiles[3])
    np_images = np.stack(tiles)
    return np_images


def plot_filters(model, model_dir):
    layer = model.layers[5]
    if type(layer) != layers.Conv2D:
        return
    filters, biases = layer.get_weights()
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    n_filters, ix = 6, 1
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        channels = f.shape[2]
        for j in range(channels):
            # specify subplot and turn of axis
            ax = pyplot.subplot(n_filters, channels, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(f[:, :, j], cmap='gray')
            ix += 1
    # save the figure
    fig1 = pyplot.gcf()
    fig1.savefig(model_dir + "/filters_L3.png")

def plot_feature_maps(model, model_dir):
    layer_output = model.layers[5].output
    # Extracts the outputs of the last conv layer
    activation_model = Model(inputs=model.input, outputs=layer_output)
    # Creates a model that will return these outputs, given the model input

    activation_model.summary()
    # load the image with the required shape
    data = load_photo('photos_used/PICT9579.png', 98)
    img = data[25,:,:,:]
    # get feature map for first hidden layer
    feature_maps = activation_model.predict(data)
    # plot all 30 maps in 3x10 squares
    num = feature_maps.shape[3]
    cols = min(6, num//5)
    rows = min(5, num//cols)
    ix = 1
    for i in range(rows):
        for j in range(cols):
            # specify subplot and turn of axis
            ax = pyplot.subplot(rows, cols, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
    # save the figure
    fig1 = pyplot.gcf()
    fig1.savefig(model_dir + "/feature_maps_L3.png")


def get_avg_error(model_num):
    err = 0
    count = 0
    filename = "predictions/predictions_model" + model_num + ".txt"
    with open(filename, "r") as file:
        for line in file:
            last = line.rstrip().split(" ")[-1]
            error = float(last)
            count += 1
            err += error
    err = err / count
    with open(filename, "a") as file:
        file.write("\n Average: {}".format(err))


if __name__ == "__main__":
    model_num = "1586869164.61565"
    get_avg_error(model_num)