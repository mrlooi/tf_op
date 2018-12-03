import torch
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDA_HOME, CUDAExtension, BuildExtension

sources = ['csrc/add_op_cpu.cpp', 'csrc/add_op_binds.cpp']
source_cuda = ['csrc/add_op_cuda.cu']
include_dirs = ['csrc']  # custom include dir here
define_macros = []
extra_compile_args = {"cxx": []}

extension = CppExtension
if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension
    sources += source_cuda
    define_macros += [("WITH_CUDA", None)]
    extra_compile_args["nvcc"] = [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]

module = extension('add_op', 
			sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,)

setup(name='add_op._C',
      ext_modules=[module],
      cmdclass={'build_ext': BuildExtension})
# setuptools.Extension(
#    name='add_op',
#    sources=['add_op.cpp'],
#    include_dirs=torch.utils.cpp_extension.include_paths(),
#    language='c++')