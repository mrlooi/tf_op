import torch
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDA_HOME, CUDAExtension, BuildExtension

sources = ['lltm.cpp']
source_cuda = []
include_dirs = []  # custom include dir here
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

module = extension('lltm', 
			sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,)

setup(name='lltm',
      ext_modules=[module],
      cmdclass={'build_ext': BuildExtension})
# setuptools.Extension(
#    name='lltm',
#    sources=['lltm.cpp'],
#    include_dirs=torch.utils.cpp_extension.include_paths(),
#    language='c++')