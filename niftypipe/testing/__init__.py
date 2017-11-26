# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
.. note:

   We use the ``nose`` testing framework for tests.

   Nose is a dependency for the tests, but should not be a dependency
   for running the algorithms in the NIPY library.  This file should
   import without nose being present on the python path.

"""

import os

# Discover directory path
filepath = os.path.abspath(__file__)
basedir = os.path.dirname(filepath)

funcfile = os.path.join(basedir, 'data', 'functional.nii')
anatfile = os.path.join(basedir, 'data', 'structural.nii')
template = funcfile
transfm = funcfile

from nose.tools import *
from numpy.testing import *

from nipype.testing import decorators as dec
from nipype.testing.utils import skip_if_no_package, package_check

skipif = dec.skipif


def example_data(infile='functional.nii'):
    """returns path to empty example data files for doc tests
    it will raise an exception if filename is not in the directory"""

    filepath = os.path.abspath(__file__)
    basedir = os.path.dirname(filepath)
    outfile = os.path.join(basedir, 'data', infile)
    if not os.path.exists(outfile):
        raise IOError('%s empty data file does NOT exist' % outfile)

    return outfile
