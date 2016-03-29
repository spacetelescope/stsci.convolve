#!/usr/bin/env python
import recon.release
from glob import glob
from numpy import get_include as np_include
from setuptools import setup, find_packages, Extension


version = recon.release.get_info()
recon.release.write_template(version, 'stsci/convolve')

setup(
    name = 'stsci.convolve',
    version = version.pep386,
    author = 'Todd Miller',
    author_email = 'help@stsci.edu',
    description = 'Image array convolution functions',
    url = 'https://github.com/spacetelescope/stsci.convolve',
    classifiers = [
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires = [
        'astropy',
        'nose',
        'numpy',
    ],
    packages = find_packages(),
    package_data = {
        '': ['LICENSE.txt']
    },
    ext_modules=[
        Extension('stsci.convolve._correlate',
            ['src/_correlatemodule.c'],
            include_dirs=[np_include()],
            define_macros=[('NUMPY','1')]),
        Extension('stsci.convolve._lineshape',
            ['src/_lineshapemodule.c'],
            include_dirs=[np_include()],
            define_macros=[('NUMPY','1')]),
    ],
)
