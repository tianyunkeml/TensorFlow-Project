import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import MNIST data here
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_dta.read_data_sets('MNIST_data/', one_hot = False)

# set tf learning parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 256
display_step = 1
examples_to_show = 10
n_input = 784

# set placeholder for graph input
X = tf.placeholder('float', [None, n_input])

# set number of neurons for each hidden layer
n_hidden_1 = 256
n_hidden_2 = 128

# set weights W and biases B
weights = {
	'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
	'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biases = {
	'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'decoder_b2': tf.Variable(tf.random_normal[n_input])
}