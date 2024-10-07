from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# class CMakeExtension(Extension):
#     def __init__(self, name: str, sourcedir: str = "") -> None:
#         super().__init__(name, sources=[])
#         self.sourcedir =""
import os 
current_directory = os.path.dirname(os.path.abspath(__file__))
print(current_directory)
setup(
    name='ggml_mm',
    ext_modules=[
        CUDAExtension(
            name='ggml_mm',
            sources=['ggml_mm.cpp'],
            include_dirs=[
                current_directory+'/ggml-master/ggml/include',
                current_directory+'/ggml-master/include',
            ],
            library_dirs=[
                current_directory+"/ggml-master/build/src",
                "/workspace/spack/opt/spack/linux-ubuntu20.04-icelake/gcc-9.4.0/cuda-12.1.1-w3vq6gjneuqhhv3okq365sci7ho2tttj/lib64",
            ],
            libraries=['ggml'],
            extra_compile_args={
            
                'nvcc': ['-std=c++11'],
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)