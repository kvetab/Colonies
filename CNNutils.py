import numpy as np
from sklearn.model_selection import train_test_split
#load input data from files

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
                self._images = self._images[perm]
                self._labels = self._labels[perm]
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
                assert batch_size <= self._num_examples
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

    class input_data:
        def __init__(self, xtrain, ytrain, xtest, ytest):
            self.train = set(xtrain, ytrain)
            self.test = set(xtest, ytest)


    data = input_data(X_train, y_train, X_test, y_test)
    return data


# define inputData.train.next_batch(batch_size=batch_size)

if __name__ == "__main__":
    data = LoadInput("outPICT9563.txt", "labPICT9563.txt")
    pokus = data.test.images
    i = 1