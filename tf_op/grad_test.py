import tensorflow as tf
import numpy as np

# x = np.ones((4,3), dtype=np.float32)
x = np.arange(12).reshape((4,3)).astype(np.float32)
x2 = x.T 
# x2 = np.arange(12).reshape((3,4)).astype(np.float32)
# x = np.ones((5,1), dtype=np.float32)
# x2 = np.arange(10).reshape((5,2)).astype(np.float32)
# x = x2.T 
np_m = np.matmul(x,x2)

X = tf.constant(x)
X2 = tf.constant(x2)

M = tf.matmul(X,X2)
# Y = tf.reduce_sum(M)

grad = tf.gradients(M, [X,X2])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
grad_value = sess.run([grad])
print(grad_value)
