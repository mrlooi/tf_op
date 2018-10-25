import os
import torch
from torch.utils.ffi import create_extension

with_cuda = torch.cuda.is_available()
print("WITH CUDA %s"%(with_cuda))

headers = ['src/add_op.h']
sources = ['src/add_op.c']
defines = []

if with_cuda:
    print('Including CUDA code.')
    headers += ['src/add_op_cuda.h']
    sources += ['src/add_op_cuda.c']
    defines += [('WITH_CUDA', None)]

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects = ['src/add_op.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
	extra_compile_args=['-std=c99'],
	name='_ext.add_op',
	headers=headers,
	sources=sources,
    define_macros=defines,
    relative_to=__file__,
    extra_objects=extra_objects,
	with_cuda=with_cuda
)

if __name__ == '__main__':
	ffi.build()
