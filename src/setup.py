from multiprocessing import cpu_count
import os
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution
from shutil import copyfile
from subprocess import check_call
from sys import platform

class SetupConfigurationError(Exception):
    pass


class CMakeBuild(build_py):
    SHLIB = "libKMCUDA.so"

    def run(self):
        if not self.dry_run:
            self._build()
        super(CMakeBuild, self).run()

    def get_outputs(self, *args, **kwargs):
        outputs = super(CMakeBuild, self).get_outputs(*args, **kwargs)
        outputs.extend(self._shared_lib)
        return outputs

    def _build(self, builddir=None):
        if platform != "darwin":
            cuda_toolkit_dir = os.getenv("CUDA_TOOLKIT_ROOT_DIR")
            cuda_arch = os.getenv("CUDA_ARCH", "61")
            if cuda_toolkit_dir is None:
                raise SetupConfigurationError(
                    "CUDA_TOOLKIT_ROOT_DIR environment variable must be defined")
            check_call(("cmake", "-DCMAKE_BUILD_TYPE=Release", "-DDISABLE_R=y",
                        "-DCUDA_TOOLKIT_ROOT_DIR=%s" % cuda_toolkit_dir,
                        "-DCUDA_ARCH=%s" % cuda_arch,
                        "."))
        else:
            ccbin = os.getenv("CUDA_HOST_COMPILER", "/usr/bin/clang")
            env = dict(os.environ)
            env.setdefault("CC", "/usr/local/opt/llvm/bin/clang")
            env.setdefault("CXX", "/usr/local/opt/llvm/bin/clang++")
            env.setdefault("LDFLAGS", "-L/usr/local/opt/llvm/lib/")
            check_call(("cmake", "-DCMAKE_BUILD_TYPE=Release", "-DDISABLE_R=y",
                        "-DCUDA_HOST_COMPILER=%s" % ccbin, "-DSUFFIX=.so", "."),
                       env=env)
        check_call(("make", "-j%d" % cpu_count()))
        self.mkpath(self.build_lib)
        dest = os.path.join(self.build_lib, self.SHLIB)
        copyfile(self.SHLIB, dest)
        self._shared_lib = [dest]


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False

setup(
    name="libKMCUDA",
    description="Accelerated K-means and K-nn on GPU",
    version="6.2.2",
    license="Apache Software License",
    author="Vadim Markovtsev",
    author_email="vadim@sourced.tech",
    url="https://github.com/src-d/kmcuda",
    download_url="https://github.com/src-d/kmcuda",
    py_modules=["libKMCUDA"],
    install_requires=["numpy"],
    distclass=BinaryDistribution,
    cmdclass={'build_py': CMakeBuild},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
	"Programming Language :: Python :: 3.6",
    ]
)

# python3 setup.py bdist_wheel
# auditwheel repair -w dist dist/*
# twine upload dist/*manylinux*
