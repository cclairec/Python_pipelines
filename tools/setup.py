#!/usr/bin/env python

from distutils.core import setup

setup(name='NiftyPipe Tools',
      version='0.1',
      description='Utilities used in niftypipe development',
      author='NiftyPipe Developers',
      author_email='nipy-devel@neuroimaging.scipy.org',
      url='http://nipy.sourceforge.net',
      scripts=['./niftypipe_nightly.py', './report_coverage.py']
      )
