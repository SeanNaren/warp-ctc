# build.py
import os

from torch.utils.ffi import create_extension

extra_compile_args = ['-std=c++11', '-fPIC']
ffi = create_extension(
    name='_ext.pytorch_ctc',
    headers=['src/binding.h'],
    sources=['src/binding.cpp'],
    with_cuda=False,
    include_dirs=[os.path.realpath('../include')])

ffi.build()
