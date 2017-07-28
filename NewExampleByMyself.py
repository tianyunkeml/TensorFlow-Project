# I want to try autoencoder with tensorflow myself

# Basic idea is that: the data is a k * k matrix, 
# with row m and column n being 1,
# and all other elements are 0:

# eg. m = 2, n = 6, then data may look like this(I used 10 * 10 for simplicity):
# 0 0 0 0 0 1 0 0 0 0
# 1 1 1 1 1 1 1 1 1 1
# 0 0 0 0 0 1 0 0 0 0
# 0 0 0 0 0 1 0 0 0 0
# 0 0 0 0 0 1 0 0 0 0
# 0 0 0 0 0 1 0 0 0 0
# 0 0 0 0 0 1 0 0 0 0
# 0 0 0 0 0 1 0 0 0 0
# 0 0 0 0 0 1 0 0 0 0
# 0 0 0 0 0 1 0 0 0 0

# and I want to encode it into 2-dimensional layer, because 2-dimensional
# data can completely decide it. Lets see the accuracy here.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import pdb

class myExample(object):
	def __init__(self):
		# parameter settings here
		self.matrixSize = 30
		self.dataSize = 90000
		self.batchSize = 300
		self.training_epochs = 10
		self.learning_rate = 0.01
		self.n_input = self.matrixSize ** 2
		self.n_hidden_1 = 600
		self.n_hidden_2 = 200

	# Generate data for feeding with batches
	def dataGenerator(self):
		allBatch = []
		for _ in range(int(self.dataSize / self.batchSize)):
			batch = []
			for i in range(self.batchSize):
				randRow = random.randint(1, self.matrixSize)
				randCol = random.randint(1, self.matrixSize)
				batch.append(self.generateOneData(randRow, randCol))
			allBatch.append(batch)
		return allBatch

	def generateOneData(self, row, col):
		data = np.zeros((self.matrixSize, self.matrixSize))
		for i in range(self.matrixSize):
			data[row - 1][i] = 1
			data[i][col - 1] = 1
		return list(data.flatten())

	def tfModel(self):
		def encoder(x):
			layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),
				self.biases['encoder_b1']))
			layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
				self.biases['encoder_b2']))
			return layer_2

		def decoder(x):
			layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']),
				self.biases['decoder_b1']))
			layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
				self.biases['decoder_b2']))
			return layer_2

		data = self.dataGenerator()

		self.weights = {
			'encoder_h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
			'encoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
			'decoder_h1': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_1])),
			'decoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_input]))
		}

		self.biases = {
			'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
			'encoder_b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
			'decoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
			'decoder_b2': tf.Variable(tf.random_normal([self.n_input]))
		}

		# X is the input matrix data
		X = tf.placeholder('float', [None, self.n_input])

		encoder_op = encoder(X)
		decoder_op = decoder(encoder_op)

		# define prediction and true value
		y_pred = decoder_op
		y_true = X

		# define lost function and optimizer
		cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
		optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

		# start session
		with tf.Session() as sess:
			init = tf.global_variables_initializer()
			sess.run(init)

			total_batch = int(self.dataSize / self.batchSize)
			for epoch in range(self.training_epochs):
				for i in range(total_batch):
					batch = data[i]
					# pdb.set_trace()
					_, c = sess.run([optimizer, cost], feed_dict = {X: batch})
				print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(c))

			# test 
			numOfPics = 10
			testData = data[0][:numOfPics]
			encode_decode = sess.run(
				y_pred, feed_dict = {X: testData})
			f, a = plt.subplots(2, 10, figsize = (10, 2))
			# pdb.set_trace()
			for i in range(numOfPics):
				a[0][i].imshow(np.reshape(testData[i], (self.matrixSize, self.matrixSize)))
				a[1][i].imshow(np.reshape(encode_decode[i], (self.matrixSize, self.matrixSize)))
			plt.show()

if __name__ == '__main__':
	myExample().tfModel()