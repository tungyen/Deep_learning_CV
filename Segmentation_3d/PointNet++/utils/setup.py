import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def get_extension_modules():
    return [
        CUDAExtension(
            name='_ball_query_cuda',
            source=[
                'ball_query/ball_query.cpp',
                'ball_query/ball_query_kernel.cu'
            ],
            include_dirs=['ball_query']
        ),
        CUDAExtension(
            name='_furthest_point_sampling_cuda',
            source=[
                'furthest_point_sampling/furthest_point_sampling.cpp',
                'furthest_point_sampling/furthest_point_sampling_kernel.cu'
            ],
            include_dirs=['furthest_point_sampling']
        ),
        CUDAExtension(
            name='_squared_distance_cuda',
            sources=[
                'squared_distance/squared_distance.cpp',
                'squared_distance/squared_distance_kernel.cu'
            ],
            include_dirs=['squared_distance']
        )
    ]
    
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6' # Compiling GPU architecture

setup(
    name='pointnet_plus_utils',
    ext_modules=get_extension_modules(),
    cmdclass={'build_ext': BuildExtension}
)