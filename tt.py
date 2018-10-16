import numpy as np
import tensorflow as tf


def test1():
  zero_out_module = tf.load_op_library('./libtf_op_test2.so')
  with tf.device('/device:GPU:0'):
  # with tf.device('/cpu:0'):
    op = zero_out_module.tf_op_test
    data = tf.Variable(np.array([[[1, 2], [3, 4]],[[4,5],[6,5]],[[4,5],[6,5]]],dtype=np.float32))  # need to use tf.Variable, otherwise tf will run on CPU as default
    opd = op(data)
  with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    x = opd.eval()
    # data = [[1, 2], [3, 4]]
    print(x)
  # Prints
  # array([[1, 0], [0, 0]], dtype=int32)

def dot_prod_demo():
  import dot_product_grad

  mm_module = tf.load_op_library('./libdot_product_op.so')
  with tf.device('/device:GPU:0'):
  # with tf.device('/cpu:0'):
    mm = mm_module.dot_product_op
    # data = tf.Variable(np.array([[1], [2]], [[1, 2], [3, 4]],dtype=np.float32))  # need to use tf.Variable, otherwise tf will run on CPU as default
    # d1 = np.array([[1, 2, 3,4], [3, 4, 5,6], [5,6,7,8]], dtype=np.float32)
    # d2 = np.array([[6,5,3,4], [2,3,6,7], [6,5,1,2]], dtype=np.float32).T
    d1 = np.arange(100).reshape((10,10)).astype(np.float32)
    # d2 = np.arange(10).reshape((2,5)).astype(np.float32)
    d2 = d1.T
    td1 = tf.Variable(d1)
    td2 = tf.Variable(d2)
    td_out = mm(td1,td2)
    grad = tf.gradients(td_out, [td1,td2])
    M = tf.matmul(td1,td2)
    grad2 = tf.gradients(M, [td1,td2])

  with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer()) 
    x, x2, g, g2 = sess.run([td_out, M, grad, grad2])
    print(x.astype(np.int32))
    print(x2.astype(np.int32))
    print(g)
    print(g2)


# test1()
dot_prod_demo()
