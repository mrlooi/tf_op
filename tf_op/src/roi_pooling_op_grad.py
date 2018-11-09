import tensorflow as tf
from tensorflow.python.framework import ops
import roi_pooling_op

@ops.RegisterShape("RoiPool")
def _roi_pool_shape(op):
  """Shape function for the RoiPool op.

  """
  try:
      dims_data = op.inputs[0].get_shape().as_list()
      channels = dims_data[3]
  except:
      channels = 3

  dims_rois = op.inputs[1].get_shape().as_list()
  num_rois = dims_rois[0]

  pooled_height = op.get_attr('pooled_height')
  pooled_width = op.get_attr('pooled_width')
  pool_channel = op.get_attr('pool_channel')

  if pool_channel == 1:
    output_shape = tf.TensorShape([num_rois, pooled_height, pooled_width, 1])
  else:
    output_shape = tf.TensorShape([num_rois, pooled_height, pooled_width, channels])
  return [output_shape, output_shape]