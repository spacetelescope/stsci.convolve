# Last Change: Wed Mar 05 09:00 PM 2008 J
from numpy.distutils.misc_util import get_numpy_include_dirs
from numpy import get_numarray_include
from numscons import GetNumpyEnvironment

env = GetNumpyEnvironment(ARGUMENTS)

env.AppendUnique(CPPPATH = [get_numpy_include_dirs(), get_numarray_include()])
env.AppendUnique(CPPDEFINES = {'NUMPY': '1'})

# _correlate extension
env.NumpyPythonExtension('_correlate', source = 'src/_correlatemodule.c') 
env.NumpyPythonExtension('_lineshape', source = 'src/_lineshapemodule.c') 
