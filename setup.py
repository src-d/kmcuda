from multiprocessing import cpu_count
import os
from shutil import copyfile
from setuptools import setup
from setuptools.command.build_py import build_py
from subprocess import check_call
from sys import platform


class CMakeBuild(build_py):
    SHLIBEXT = "dylib" if platform == "darwin" else "so"

    def run(self):
        if not self.dry_run:
            self._build()
        super(CMakeBuild, self).run()

    def get_outputs(self, *args, **kwargs):
        outputs = super(CMakeBuild, self).get_outputs(*args, **kwargs)
        outputs.extend(self._shared_lib)
        return outputs

    def _build(self, builddir=None):
        check_call(("cmake", "-DCMAKE_BUILD_TYPE=Release", "."))
        check_call(("make", "-j%d" % cpu_count()))
        self.mkpath(self.build_lib)
        shlib = "libKMCUDA." + self.SHLIBEXT
        dest = os.path.join(self.build_lib, shlib)
        copyfile(shlib, dest)
        self._shared_lib = [dest]


setup(
    name="libKMCUDA",
    description="Accelerated K-means on GPU",
    version="2.0.0",
    license="MIT",
    author="Vadim Markovtsev",
    author_email="vadim@sourced.tech",
    url="https://github.com/src-d/kmcuda",
    download_url="https://github.com/src-d/kmcuda",
    py_modules=["libKMCUDA"],
    install_requires=["numpy"],
    cmdclass={'build_py': CMakeBuild},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5"
    ]
)