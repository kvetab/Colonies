from bin import CNN
import tensorflow as tf

output_fctions = (tf.nn.relu, tf.nn.sigmoid, CNN.sigmoid_shifted)
sigmoid_functions = (tf.nn.sigmoid, CNN.sigmoid_shifted)
filter_nums = ((6, 10, 16), (10, 20, 30), (16, 30, 40))
learning_rate = 0.001
epochs = 40
batch_size = 16
split_filters = (True, False)
what_to_sum = ((0, 0, 1), (0, 1, 1), (1, 1, 1))


def TestAll():
    for fction_l in output_fctions:
        for fction_s in output_fctions:     # sem mozna jen sigmoidni funkce?
            for num in filter_nums:
                for split in split_filters:
                    for sums in what_to_sum:
                        print()
                        print("Testing: outf_l = ", fction_l.__name__, "; outf_s = ", fction_s.__name__, "; filter num = ", str(num), "; split filters = ", split, "; what to sum:", sums)
                        CNN.run_cnn(learning_rate, epochs, batch_size, fction_l, fction_s, num, split, sums)

def TestSome():
    for num in filter_nums:
        for sums in what_to_sum:
            for split in split_filters:
                print()
                print("Testing: filter num = ", str(num), "; split filters = ", split, "; what to sum:", sums)
                CNN.run_cnn(learning_rate, epochs, batch_size, tf.nn.sigmoid, tf.nn.sigmoid, num, split, sums)

def testFilters():
    filter_nums_ = ((1, 2, 3), (5, 10, 15), (10, 20, 30), (15, 30, 45))
    for nums in filter_nums_:
        print()
        print("Testing " + str(nums) + " filters")
        CNN.run_cnn(learning_rate, epochs, batch_size, tf.nn.sigmoid, tf.nn.sigmoid, nums, False, (0, 0, 1))

TestSome()
