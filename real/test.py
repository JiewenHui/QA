import tensorflow as tf
import numpy as np
a = np.arange(100).reshape(10,10) + 0.0

b = tf.norm(a,ord = 'euclidean',axis = 1)
with tf.Session() as sess:
	print sess.run(b)
bound = -0.0075
a = np.arange(10)
rng = np.random.RandomState(23455)
print rng.uniform(low = -bound,high = bound,size = [10])

