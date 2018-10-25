import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
dot_product_grad_module = tf.load_op_library('./libdot_product_grad_op.so')

OP_NAME = "DotProductOp"

@ops.RegisterGradient(OP_NAME) # MUST BE SAME AS OP NAME
def _dot_product_grad_cc(op, grad):
    """
    The gradient for `dot_product` using the operation implemented in C++.
    
    :param op: `dot_product` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `dot_product` op.
    :return: gradients with respect to the input of `dot_product`.
    """
    return dot_product_grad_module.dot_product_grad_op(grad, op.inputs[0], op.inputs[1])  # METHOD NAME must be the snake case of the Op Name, as defined in the module

# # uncomment this and comment the corresponding line above to use the Python
# # implementation of the dot product gradient
# #@ops.RegisterGradient(OP_NAME)
# def _dot_product_grad(op, grad):
  
#     input_tensor = op.inputs[0]
#     weight_tensor = op.inputs[1]
#     input_rows = array_ops.shape(input_tensor)[0]
#     output_rows = array_ops.shape(weight_tensor)[0]
    
#     grad_input = tf.matmul(tf.transpose(grad), weight_tensor)
#     grad_weights = tf.multiply(tf.transpose(grad), tf.reshape(tf.tile(tf.reshape(input_tensor, [input_rows]), [output_rows]), [output_rows, -1]))
    
#     return [tf.transpose(grad_input), grad_weights]
