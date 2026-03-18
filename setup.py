import os
from glob import glob
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="CoDec",
    version="0.1.0",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "test",
            "docs",
            "benchmark",
            "codec.egg-info",
        )
    ),
    description="CoDec: Prefix-Shared Decoding Kernel for LLMs",
    url="https://github.com/wzbxpy/codec",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=[
        CUDAExtension(
            name="codec_gpu",  # Same as TORCH_EXTENSION_NAME
            sources=[
                *glob("csrc/*.cpp"),
                *glob("csrc/kernel/*.cu"),
            ],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++20",
                ],
                "nvcc": [
                    "-O3",
                    "-std=c++20",
                    "--threads=0",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                ],
            },
            include_dirs=[
                Path(this_dir) / "csrc" / "cutlass" / "include",
                Path(this_dir) / "csrc" / "include",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch",
        "einops",
    ],
    setup_requires=[
        "ninja",
    ],
)
