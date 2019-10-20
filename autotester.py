import CNN
import tensorflow as tf
import numpy as np

output_fctions = (tf.nn.relu, tf.nn.sigmoid, CNN.sigmoid_shifted)
sigmoid_functions = (tf.nn.sigmoid, CNN.sigmoid_shifted)
filter_nums = ((6, 10, 16), (10, 20, 30), (16, 30, 40))
learning_rate = 0.0001
epochs = 10
batch_size = 16
split_filters = (True, False)
what_to_sum = ((0, 0, 1), (0, 1, 1), (1, 1, 1))


for fction_l in output_fctions:
    for fction_s in output_fctions:     # sem mozna je sigmoidni funkce?
        for num in filter_nums:
            for split in split_filters:
                for sums in what_to_sum:
                    print()
                    print("Testing: outf_l = ", fction_l.__name__, "; outf_s = ", fction_s.__name__, "; filter num = ", str(num), "; split filters = ", split, "; what to sum:", sums)
                    CNN.run_cnn(learning_rate, epochs, batch_size, fction_l, fction_s, num, split, sums)
