import CNNkeras
from CNNkeras import sigmoid_ext, sigmoid_shifted
import tensorflow as tf

output_fctions = (tf.nn.relu, tf.nn.sigmoid, CNNkeras.sigmoid_shifted)
sigmoid_functions = (CNNkeras.sigmoid_shifted,  CNNkeras.sigmoid_ext, tf.nn.sigmoid)
filter_nums = ((6, 10, 16), (10, 20, 30), (16, 30, 40))
learning_rate = 0.0001
epochs = 160
batch_size = 16
split_filters = (True, False)
what_to_sum = ((0, 0, 1), (0, 1, 1), (1, 1, 1))


# CNNkeras.pipeline(0.0001,200,16,tf.keras.activations.relu,tf.keras.activations.sigmoid,(16, 30, 40), False,(0, 0, 1), True, True)
CNNkeras.pipeline(0.0001,200,16,tf.keras.activations.relu,sigmoid_ext,(16, 30, 40), False,(0, 1, 1), "/mnt/0/models/dodatek/", False, True)
CNNkeras.pipeline(0.0001,200,16,tf.keras.activations.relu,sigmoid_ext,(6, 10, 16), False,(1, 1, 1), "/mnt/0/models/dodatek/", True, True)
CNNkeras.pipeline(0.0001, 200, 16, tf.keras.activations.relu, sigmoid_ext, (16, 30, 40), False,(0, 0, 1), "/mnt/0/models/dodatek/", False, False)

CNNkeras.pipeline(0.0001,200,16,tf.keras.activations.relu,tf.keras.activations.sigmoid,(16, 30, 40), False,(0, 1, 1), "/mnt/0/models/dodatek/", False, True)
CNNkeras.pipeline(0.0001,200,16,tf.keras.activations.relu,sigmoid_shifted,(16, 30, 40), False,(0, 1, 1), "/mnt/0/models/dodatek/", False, True)
CNNkeras.pipeline(0.0001,200,16,tf.keras.activations.relu,sigmoid_ext,(16, 30, 40),False, (0, 0, 1), "/mnt/0/models/dodatek/", True, True)
CNNkeras.pipeline(0.0001,200,16,tf.keras.activations.relu,sigmoid_shifted,(16, 30, 40), True,(0, 1, 1), "/mnt/0/models/dodatek/", True, True)





