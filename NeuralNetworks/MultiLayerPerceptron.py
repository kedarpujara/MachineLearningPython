import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 

# Dataset (numbers from 1-10)
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
n_samples = mnist.train.num_examples


# Establishing learning rates and epochs
learning_rate = 0.001
training_epochs = 15 
batch_size = 100

# Establish the values for hidden layers. Input is 784 because images are 28x28 pixels
# and we want to output 10 classes, representing the numbers 1-10
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10


# Establish the weights between each layer
weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))

}


# Establish the biases between each layer
biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_classes]))
}

# Core of the actual Multilayer Perceptron 
def MLP():
	x = tf.placeholder("float",  [None, n_input])
	y = tf.placeholder("float", [None, n_classes])

	pred = MLPLayerSetup(x, weights, biases)

	# Setting up the cost function
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


	x_samp, y_samp = mnist.train.next_batch(1)
	plt.imshow(x_samp.reshape(28,28))
	print(y_samp)

	plt.show()


	#Running the Session 
	init = tf.initialize_all_variables()
	sess = tf.InteractiveSession()

	sess.run(init)

	for epoch in range(training_epochs):

		avg_cost = 0.0
		total_batch = int(n_samples/batch_size)


		for i in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			_,c = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y})

			avg_cost += c/total_batch
   		print("Epoch: {} cost={:.4f}".format(epoch+1,avg_cost))

   	print("Model has completed {} Epochs of Training".format(training_epochs))


   	#Evaluations
   	correct_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
   	print(correct_predictions[0])

   	correct_predictions = tf.cast(correct_predictions, "float")
   	print(correct_predictions[0])

   	accuracy = tf.reduce_mean(correct_predictions)
   	print("Accuracy: ", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))



# Setup for the actual perceptron with all three layers, weights, and biases
def MLPLayerSetup(x, weights, biases):
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)

	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)

	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return out_layer




MLP()
