from distutils.core import setup, Extension
import os
import numpy

H2PACK_DIR = "../src"

extra_cflags = ["-I"+H2PACK_DIR+"/include"]
extra_cflags += ["-g", "-std=gnu99", "-O3", "-DUSE_MKL"]
extra_cflags += ["-qopenmp", "-xHost", "-mkl"]

LIB = [H2PACK_DIR+"/libH2Pack.a"]
extra_lflags = LIB + ["-g", "-O3", "-qopenmp", "-L${MKLROOT}/lib/intel64", "-mkl_rt", "-lpthread"]

def main():
    setup(name="pyh2pack",
        version="1.0.0",
        description="Python interface for H2Pack",
        author="Hua Huang, and Xin Xing",
        author_email="xxing02@gmail.com",
        ext_modules=[Extension(
            name = "pyh2pack",
            sources = ["pyh2pack.c"],
            include_dirs=["../src/include", numpy.get_include()],
            extra_compile_args = extra_cflags,
            extra_link_args= extra_lflags,
            )
        ]
    )

if __name__ == "__main__":
    main()
