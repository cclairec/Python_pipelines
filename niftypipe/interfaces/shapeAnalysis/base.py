# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
The Shape Analysis module provides classes for interfacing with `Claicury stuff'

Examples
--------
See the docstrings of the individual classes for examples.
"""

from nipype.interfaces.base import CommandLine, BaseInterfaceInputSpec, traits
from nipype.interfaces.matlab import MatlabInputSpec
from nipype.utils.filemanip import split_filename
import os
import warnings


warn = warnings.warn
warnings.filterwarnings('always', category=UserWarning)


def get_shape_path(env='SHAPE_PATH'):
    """Check if shape is installed."""
    return os.environ.get(env, '')


def no_shape():
    """Check if shape is installed."""
    if get_shape_path():
        return False
    return True


def no_deformetrica(cmd='sparseMatching3'):
    """Check if Deformetrica is installed."""
    if True in [os.path.isfile(os.path.join(path, cmd)) and
                os.access(os.path.join(path, cmd), os.X_OK)
                for path in os.environ["PATH"].split(os.pathsep)]:
        return False
    return True


class DeformetricaCommand(CommandLine):
    """
    Base support interface for NiftySeg commands.
    """
    _suffix = '_deformetrica'

    def __init__(self, **inputs):
        super(DeformetricaCommand, self).__init__(**inputs)

    def _gen_fname(self, basename, out_dir=None, suffix=None, ext=None):
        if basename == '':
            msg = 'Unable to generate filename for command %s. ' % self.cmd
            msg += 'basename is not set!'
            raise ValueError(msg)
        _, final_bn, final_ext = split_filename(basename)
        if out_dir is None:
            out_dir = os.getcwd()
        if ext is not None:
            final_ext = ext
        if suffix is not None:
            final_bn = ''.join((final_bn, suffix))
        return os.path.abspath(os.path.join(out_dir, final_bn + final_ext))


class ShapeInputSpec(BaseInterfaceInputSpec):
    """
    Base Class for inputs used by Shape Interfaces.
    """
    desc = 'Folder containing the matlab_startup.m defining Shape matlab paths to add from your local computer. \
By default, using matlab_startup.m from path defined by SHAPE_PATH from environment variables.'
    path_matlab = traits.Str(get_shape_path(), desc=desc, usedefault=True)


class ShapeMatlabInputSpec(MatlabInputSpec):
    """
    Base Class for inputs used by Shape Interfaces.
    """
    desc = 'Folder containing the matlab_startup.m defining Shape matlab paths to add from your local computer. \
By default, using matlab_startup.m from path defined by SHAPE_PATH from environment variables.'
    path_matlab = traits.Str(get_shape_path(), desc=desc, usedefault=True)
