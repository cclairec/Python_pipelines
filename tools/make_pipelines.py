#!/usr/bin/env python
"""Run the py->rst conversion and run all pipelines.

This also creates the index.rst file appropriately, makes figures, etc.
"""

from past.builtins import execfile
# -----------------------------------------------------------------------------
# Library imports
# -----------------------------------------------------------------------------

# Stdlib imports
import os
import sys

from glob import glob

# Third-party imports

# We must configure the mpl backend before making any further mpl imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib._pylab_helpers import Gcf

# Local tools
from toollib import *

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------

pipelines_header = """

.. _pipelines:

Pipelines
========

.. note_about_pipelines
"""
# -----------------------------------------------------------------------------
# Function defintions
# -----------------------------------------------------------------------------

# These global variables let show() be called by the scripts in the usual
# manner, but when generating pipelines, we override it to write the figures to
# files with a known name (derived from the script name) plus a counter
figure_basename = None

# We must change the show command to save instead


def show():
    allfm = Gcf.get_all_fig_managers()
    for fcount, fm in enumerate(allfm):
        fm.canvas.figure.savefig('%s_%02i.png' %
                                 (figure_basename, fcount + 1))

_mpl_show = plt.show
plt.show = show

# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

# Work in pipelines directory
cd('users/pipelines')
if not os.getcwd().endswith('users/pipelines'):
    raise OSError('This must be run from doc/pipelines directory')

# Run the conversion from .py to rst file
sh('../../../tools/ex2rst --project NiftyPipe --outdir . ../../../niftypipe/bin')

# Make the index.rst file
"""
index = open('index.rst', 'w')
index.write(pipelines_header)
for name in [os.path.splitext(f)[0] for f in glob('*.rst')]:
    #Don't add the index in there to avoid sphinx errors and don't add the
    #note_about pipelines again (because it was added at the top):
    if name not in(['index','note_about_pipelines']):
        index.write('   %s\n' % name)
index.close()
"""

# Execute each python script in the directory.
if '--no-exec' in sys.argv:
    pass
else:
    if not os.path.isdir('fig'):
        os.mkdir('fig')

    for script in glob('*.py'):
        figure_basename = pjoin('fig', os.path.splitext(script)[0])
        execfile(script)
        plt.close('all')
