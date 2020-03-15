import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
#load input data from files

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
        images.append(numpy_img[:, :, 0:3])  # LP:hack to get rid of alpha in case of RGBA
        # print(numpy_img.shape)
    np_images = np.stack(images)
    np_labels = labels.label.to_numpy(copy=True)
    X_train, X_test, y_train, y_test = train_test_split(np_images, np_labels, test_size = 0.1, random_state = 42)
    return X_train, X_test, y_train, y_test


# takes all image filenames from label file and loads the images
# splits images and labels into test and train sets
# returns input_data structure, which has next_batch function defined
def LoadInputIMG(file_labels):
    labels = pd.read_csv(file_labels, header=None)
    labels.columns = ["img","label"]
    #print(labels.head())

    images = []
    for i in labels.img:
        numpy_img = load_image("male/crops/"+i)
        images.append(numpy_img[:,:,0:3])   # LP:hack to get rid of alpha in case of RGBA
        #print(numpy_img.shape)
    np_images = np.stack(images)
    #print(np_images.shape)

    np_labels = labels.label.to_numpy(copy=True)
    X_train, X_test, y_train, y_test = train_test_split(np_images, np_labels, test_size = 0.1, random_state = 42)
    #test = {"images": X_test, "labels": y_test}
    #train = {"images": X_train, "labels": y_train}
    #data = {"train": train, "test": test}

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



# previous version with text data instead of images
def LoadInput(file_data, file_labels):
    data = np.genfromtxt(file_data, delimiter=",", skip_header=0)
    labels = np.genfromtxt(file_labels, delimiter=",", skip_header=0)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3, random_state = 42)
    #test = {"images": X_test, "labels": y_test}
    #train = {"images": X_train, "labels": y_train}
    #data = {"train": train, "test": test}

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


# define inputData.train.next_batch(batch_size=batch_size)

if __name__ == "__main__":
    data = LoadInput("outPICT9563.txt", "labPICT9563.txt")
    pokus = data.test.images
    i = 1