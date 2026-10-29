from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name= "codegemm_cpu_kernel",
    version= "0.1.0",
    ext_modules= [
        CppExtension(
            name= "codegemm_cpu_kernel",
            sources= ["kernels.cpp"],
            extra_compile_args= ["-O2", "-g0"],
        ),
    ],
    cmdclass= {"build_ext": BuildExtension},
)