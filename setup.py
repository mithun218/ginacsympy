# This Python file uses the following encoding: utf-8

from setuptools import Extension,setup,find_packages
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import os
import re
import shutil
import sys


class Build(build_ext):
    # def build_extensions(self):
    #     if self.compiler.compiler_type == 'mingw32':
    #         for e in self.extensions:
    #             e.extra_link_args = gcc_lflags
    #     else:
    #         for e in self.extensions:
    #             e.extra_link_args = gcc_lflags
    #     super(Build, self).build_extensions()

    #Including necessary files with GinacSympy package
#    def find_package_modules(self, package, package_dir):
#            modules = super().find_package_modules(package, package_dir)
#            return [(pkg, mod, file, ) for (pkg, mod, file, ) in modules if mod == 'ginacsympy_version']

# The overwriting of the build_ext class run method was needed to get the ginacsympy_version.py file inside the wheel.
    def run(self):
        build_ext.run(self)
        build_dir = os.path.realpath(self.build_lib)
        root_dir = os.path.dirname(os.path.realpath(__file__))
        target_dir = build_dir if not self.inplace else root_dir
        self.copy_file('ginacsympy_version.py', root_dir, target_dir)
        self.copy_file('ginacsympy_abc.py', root_dir, target_dir)

    def copy_file(self, path, source_dir, destination_dir):
        if os.path.exists(os.path.join(source_dir, path)):
            shutil.copyfile(os.path.join(source_dir, path),
                    os.path.join(destination_dir, path))


extensions = [
    Extension("ginacsympy", ["ginacsympy.pyx"],
        libraries=["ginacsym","flint","cln","mpfr","gmp"],
        #libraries=["ginacsymWithClnFlint","cln","gmp","mpfr","flint"],
        # libraries=["ginacsymstatic"],  
        # library_dirs=["F:\msys64\mingw64\lib"],
        # extra_compile_args=['-ldl'],
        # extra_link_args=["-L."],
        )
]

def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(project + 'ginacsympy_version.py').read())
    return result.group(1)

setup(
    name = "ginacsympy",
    version = get_property('__version__', ""),
    author = "Mithun Bairagi",
    author_email = "bairagirasulpur@gmail.com",
    description = "A Cython frontend to the fast C++ symbolic manipulation library GinacSym. ",
    license='GPLv2 or above',
    url = "https://htmlpreview.github.io/?https://github.com/mithun218/ginacsympy/blob/master/doc/html/index.html",
    project_urls={
              'Source': 'https://github.com/mithun218/ginacsympy',
          },
#    packages=find_packages(), #automatically find the packages that are recognized in the __init__.py
    platforms=sys.platform,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        ],
    setup_requires = "wheel",
    python_requires='>=3.8',
    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': Build}
)

