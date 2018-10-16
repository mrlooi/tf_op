cmake_minimum_required(VERSION 2.8)

find_package(CUDA QUIET REQUIRED)


set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp ${CMAKE_CXX_FLAGS}")
# if GCC > 5
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 ${CMAKE_CXX_FLAGS}")
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 --expt-relaxed-constexpr)

# get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)
execute_process(COMMAND python -c "import tensorflow; print(tensorflow.sysconfig.get_lib())" OUTPUT_VARIABLE Tensorflow_LIBRARY_DIRS)
set(Tensorflow_LIBRARIES "tensorflow_framework")

# print
message(${Tensorflow_INCLUDE_DIRS})
message(${Tensorflow_LIBRARY_DIRS})

# includes
include_directories(include)
include_directories(${Tensorflow_INCLUDE_DIRS})

# links
link_directories("/usr/local/lib/python2.7/dist-packages/tensorflow")
link_libraries(${Tensorflow_LIBRARIES})

# BUILD LIBS
add_library(tf_op_test1 SHARED src/tf_op_test1.cc)
cuda_add_library(tf_op_test2_cu SHARED src/tf_op_test2.cu)  # output libtf_op_test2_cu.so
#set_target_properties(tf_op_test2_cu PROPERTIES OUTPUT_NAME "tf_op_test2.cu")
add_library(tf_op_test2 SHARED src/tf_op_test2.cc)
target_link_libraries(tf_op_test2 tf_op_test2_cu)


add_library(dot_product_op SHARED src/dot_product_op.cc)
cuda_add_library(dot_product_op_cu SHARED src/dot_product_op.cu) 
target_link_libraries(dot_product_op dot_product_op_cu)

add_library(dot_product_grad_op SHARED src/dot_product_grad_op.cc)
target_link_libraries(dot_product_grad_op dot_product_op_cu)