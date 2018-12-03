#include "add_op.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &add_op_forward, "add_op_forward");
  m.def("backward", &add_op_backward, "add_op_backward");
}
