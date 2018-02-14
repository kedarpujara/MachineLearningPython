import tensorflow as tf 

def tf():
	hello = tf.constant('Hello')
	sess = tf.Session()
	print(sess.run(hello))

tf()