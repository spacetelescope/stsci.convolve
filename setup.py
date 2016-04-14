#!/usr/bin/env python
import os
import subprocess
import sys
from glob import glob
from numpy import get_include as np_include
from setuptools import setup, find_packages, Extension


if os.path.exists('relic'):
    sys.path.insert(1, 'relic')
    import relic.release
else:
    try:
        import relic.release
    except ImportError:
        try:
            subprocess.check_call(['git', 'clone',
                'https://github.com/jhunkeler/relic.git'])
            sys.path.insert(1, 'relic')
            import relic.release
        except subprocess.CalledProcessError as e:
            print(e)
            exit(1)


version = relic.release.get_info()
relic.release.write_template(version, 'stsci/convolve')

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
