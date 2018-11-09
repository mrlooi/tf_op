
import tensorflow as tf
import numpy as np
import roi_pooling_op

op = roi_pooling_op.roi_pool([], [], 7, 6, 1.0/3, 0)

init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)
sess.run(op)