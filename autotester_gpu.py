import CNNkeras
import tensorflow as tf
import numpy as np

output_fctions = (tf.nn.relu, tf.nn.sigmoid, CNNkeras.sigmoid_shifted)
sigmoid_functions = (CNNkeras.sigmoid_shifted,  CNNkeras.sigmoid_ext, tf.nn.sigmoid)
filter_nums = ((6, 10, 16), (10, 20, 30), (16, 30, 40))
learning_rate = 0.0001
epochs = 160
batch_size = 16
split_filters = (True, False)
what_to_sum = ((0, 0, 1), (0, 1, 1), (1, 1, 1))




CNNkeras.pipeline(0.0001,200,16,tf.keras.activations.relu,tf.keras.activations.sigmoid,(16, 30, 40),False, (0, 0, 1))
CNNkeras.pipeline(0.0001,160,16,tf.keras.activations.relu,tf.keras.activations.sigmoid,(16, 30, 40), True,(0, 1, 1))
CNNkeras.pipeline(0.0001,200,16,tf.keras.activations.relu,tf.keras.activations.sigmoid,(6, 10, 16), True,(1, 1, 1))
CNNkeras.pipeline(0.0001,200,16,tf.keras.activations.relu,CNNkeras.sigmoid_ext,(10, 20, 30), True, (0, 1, 1))
CNNkeras.pipeline(0.0001,200,16,tf.keras.activations.relu,CNNkeras.sigmoid_ext,(10, 20, 30),False,  (0, 0, 1))
CNNkeras.pipeline(0.0001,200,16,tf.keras.activations.relu,CNNkeras.sigmoid_shifted,(6, 10, 16),False,  (0, 0, 1))

# pipeline(learning_rate, epochs, batch_size, outf_layer, outf_sum, filter_numbers, split_filters, what_to_sum, "male")


