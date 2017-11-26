#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""niftypipe : Neuroimaging in Python pipelines and interfaces package.

niftypipe intends to create python interfaces to other neuroimaging
packages and create an API for specifying a full analysis pipeline in
python.

Much of the machinery at the beginning of this file has been copied over from
nibabel denoted by ## START - COPIED FROM NIBABEL and a corresponding ## END

"""
"""Build helper."""

import os
from glob import glob
import sys
from functools import partial

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

# For some commands, use setuptools.
if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist', 'bdist_dumb',
            'install_egg_info', 'egg_info', 'easy_install', 'bdist_wheel',
            'bdist_mpkg')).intersection(sys.argv)) > 0:
    # setup_egg imports setuptools setup, thus monkeypatching distutils.
    import setup_egg

from distutils.core import setup

# Commit hash writing, and dependency checking
''' Distutils / setuptools helpers from nibabel.nisext'''

import os
from os.path import join as pjoin
import sys
PY3 = sys.version_info[0] >= 3
if PY3:
    string_types = str,
else:
    string_types = basestring,
try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser

from distutils.version import LooseVersion
from distutils.command.build_py import build_py
from distutils import log

# Attempt to import setuptools, which will allow automatic installation of dependencies
try:
    __import__('setuptools')
except ImportError:
    print('missing setuptools, advise when packages are missing')


def get_comrec_build(pkg_dir, build_cmd=build_py):
    """ Return extended build command class for recording commit

    The extended command tries to run git to find the current commit, getting
    the empty string if it fails.  It then writes the commit hash into a file
    in the `pkg_dir` path, named ``COMMIT_INFO.txt``.

    In due course this information can be used by the package after it is
    installed, to tell you what commit it was installed from if known.

    To make use of this system, you need a package with a COMMIT_INFO.txt file -
    e.g. ``myproject/COMMIT_INFO.txt`` - that might well look like this::

        # This is an ini file that may contain information about the code state
        [commit hash]
        # The line below may contain a valid hash if it has been substituted during 'git archive'
        archive_subst_hash=$Format:%h$
        # This line may be modified by the install process
        install_hash=

    The COMMIT_INFO file above is also designed to be used with git substitution
    - so you probably also want a ``.gitattributes`` file in the root directory
    of your working tree that contains something like this::

       myproject/COMMIT_INFO.txt export-subst

    That will cause the ``COMMIT_INFO.txt`` file to get filled in by ``git
    archive`` - useful in case someone makes such an archive - for example with
    via the github 'download source' button.

    Although all the above will work as is, you might consider having something
    like a ``get_info()`` function in your package to display the commit
    information at the terminal.  See the ``pkg_info.py`` module in the nipy
    package for an example.
    """
    class MyBuildPy(build_cmd):
        ''' Subclass to write commit data into installation tree '''
        def run(self):
            build_cmd.run(self)
            import subprocess
            proc = subprocess.Popen('git rev-parse --short HEAD',
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    shell=True)
            repo_commit, _ = proc.communicate()
            # Fix for python 3
            repo_commit = str(repo_commit)
            # We write the installation commit even if it's empty
            cfg_parser = ConfigParser()
            cfg_parser.read(pjoin(pkg_dir, 'COMMIT_INFO.txt'))
            cfg_parser.set('commit hash', 'install_hash', repo_commit)
            out_pth = pjoin(self.build_lib, pkg_dir, 'COMMIT_INFO.txt')
            cfg_parser.write(open(out_pth, 'wt'))
    return MyBuildPy


def _add_append_key(in_dict, key, value):
    """ Helper for appending dependencies to setuptools args """
    # If in_dict[key] does not exist, create it
    # If in_dict[key] is a string, make it len 1 list of strings
    # Append value to in_dict[key] list
    if key not in in_dict:
        in_dict[key] = []
    elif isinstance(in_dict[key], string_types):
        in_dict[key] = [in_dict[key]]
    in_dict[key].append(value)


# Dependency checks
def package_check(pkg_name,
                  version=None,
                  optional=False,
                  checker=LooseVersion,
                  version_getter=None,
                  messages=None,
                  setuptools_args=None
                  ):
    ''' Check if package `pkg_name` is present and has good enough version

    Has two modes of operation.  If `setuptools_args` is None (the default),
    raise an error for missing non-optional dependencies and log warnings for
    missing optional dependencies.  If `setuptools_args` is a dict, then fill
    ``install_requires`` key value with any missing non-optional dependencies,
    and the ``extras_requires`` key value with optional dependencies.

    This allows us to work with and without setuptools.  It also means we can
    check for packages that have not been installed with setuptools to avoid
    installing them again.

    Parameters
    ----------
    pkg_name : str
       name of package as imported into python
    version : {None, str}, optional
       minimum version of the package that we require. If None, we don't
       check the version.  Default is None
    optional : bool or str, optional
       If ``bool(optional)`` is False, raise error for absent package or wrong
       version; otherwise warn.  If ``setuptools_args`` is not None, and
       ``bool(optional)`` is not False, then `optional` should be a string
       giving the feature name for the ``extras_require`` argument to setup.
    checker : callable, optional
       callable with which to return comparable thing from version
       string.  Default is ``distutils.version.LooseVersion``
    version_getter : {None, callable}:
       Callable that takes `pkg_name` as argument, and returns the
       package version string - as in::

          ``version = version_getter(pkg_name)``

       If None, equivalent to::

          mod = __import__(pkg_name); version = mod.__version__``
    messages : None or dict, optional
       dictionary giving output messages
    setuptools_args : None or dict
       If None, raise errors / warnings for missing non-optional / optional
       dependencies.  If dict fill key values ``install_requires`` and
       ``extras_require`` for non-optional and optional dependencies.
    '''
    setuptools_mode = setuptools_args is not None
    optional_tf = bool(optional)
    if version_getter is None:
        def version_getter(pkg_name):
            mod = __import__(pkg_name)
            return mod.__version__
    if messages is None:
        messages = {}
    msgs = {
        'missing': 'Cannot import package "%s" - is it installed?',
        'missing opt': 'Missing optional package "%s"',
        'opt suffix': '; you may get run-time errors',
        'version too old': 'You have version %s of package "%s"'
        ' but we need version >= %s', }
    msgs.update(messages)
    status, have_version = _package_status(pkg_name,
                                           version,
                                           version_getter,
                                           checker)
    print(pkg_name)
    print(status)
    print(have_version)
    print('===============')
    if status == 'satisfied':
        return
    if not setuptools_mode:
        if status == 'missing':
            if not optional_tf:
                raise RuntimeError(msgs['missing'] % pkg_name)
            log.warn(msgs['missing opt'] % pkg_name +
                     msgs['opt suffix'])
            return
        elif status == 'no-version':
            raise RuntimeError('Cannot find version for %s' % pkg_name)
        assert status == 'low-version'
        if not optional_tf:
            raise RuntimeError(msgs['version too old'] % (have_version,
                                                          pkg_name,
                                                          version))
        log.warn(msgs['version too old'] % (have_version,
                                            pkg_name,
                                            version) +
                 msgs['opt suffix'])
        return
    # setuptools mode
    if optional_tf and not isinstance(optional, string_types):
        raise RuntimeError('Not-False optional arg should be string')
    dependency = pkg_name
    if version:
        dependency += '>=' + version
    if optional_tf:
        if 'extras_require' not in setuptools_args:
            setuptools_args['extras_require'] = {}
        _add_append_key(setuptools_args['extras_require'],
                        optional,
                        dependency)
        return
    # add_append_key(setuptools_args, 'install_requires', dependency)
    return


def _package_status(pkg_name, version, version_getter, checker):
    try:
        __import__(pkg_name)
    except ImportError:
        return 'missing', None
    if not version:
        return 'satisfied', None
    try:
        have_version = version_getter(pkg_name)
    except AttributeError:
        return 'no-version', None
    if checker(have_version) < checker(version):
        return 'low-version', have_version
    return 'satisfied', have_version


def main(**extra_args):
    setup(name=NAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          url=URL,
          download_url=DOWNLOAD_URL,
          license=LICENSE,
          classifiers=CLASSIFIERS,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          platforms=PLATFORMS,
          version=VERSION,
          install_requires=REQUIRES,
          provides=PROVIDES,
          packages=['niftypipe',
                    'niftypipe.interfaces',
                    'niftypipe.interfaces.niftk',
                    'niftypipe.interfaces.niftk.labels',
                    'niftypipe.interfaces.shapeAnalysis',
                    'niftypipe.workflows',
                    'niftypipe.workflows.asl',
                    'niftypipe.workflows.dmri',
                    'niftypipe.workflows.misc',
                    'niftypipe.workflows.petmri',
                    'niftypipe.workflows.rsfmri',
                    'niftypipe.workflows.shapeanalysis',
                    'niftypipe.workflows.structuralmri'],
          # Intentionally, we do NOT install the perform script that way. It is up to the user to
          # include the relative source path in their PATH env
          scripts=glob(os.path.join(NAME, 'bin', '*')),
          cmdclass=cmdclass,
          dependency_links=['https://github.com/NifTK/nipype/tarball/master#egg=nipype-'+NIPYPE_MIN_VERSION],
          **extra_args
          )
          
    # Adding flag for matplotlib for mac if version>1.4.3:
    from sys import platform as _platform
    if _platform == "darwin":
        # If version > 1.4.3, add the TkAgg tag
        try:
            import matplotlib
            vers = int(matplotlib.__version__[:5].replace('.', ''))
            if vers > 143:
                matplotlibrc_path = os.path.join(os.environ['HOME'],
                                                 '.matplotlib',
                                                 'matplotlibrc')
                if os.path.exists(matplotlibrc_path):
                    if 'backend: TkAgg' not in open(matplotlibrc_path).read():
                        with open(matplotlibrc_path, "a") as matplotlibrc:
                            # OS X: fix the backend for newest version
                            matplotlibrc.write("backend: TkAgg")
                else:
                    if not os.path.exists(os.path.dirname(matplotlibrc_path)):
                        os.makedirs(os.path.dirname(matplotlibrc_path))
                    with open(matplotlibrc_path, "w") as matplotlibrc:
                        # OS X: fix the backend for newest version
                        matplotlibrc.write("backend: TkAgg")
        except:
            print 'Error: matplotlib not found.'

if __name__ == "__main__":

    cmdclass = {'build_py': get_comrec_build('niftypipe')}

    # Get version and release info, which is all stored in niftypipe/info.py
    ver_file = os.path.join('niftypipe', 'info.py')
    exec(open(ver_file).read())

    # Prepare setuptools args
    if 'setuptools' in sys.modules:
        extra_setuptools_args = dict(
            tests_require=['nose'],
            test_suite='nose.collector',
            zip_safe=False,
            extras_require=dict(
                doc='Sphinx>=0.3',
                test='nose>=0.10.1'),
        )
    else:
        extra_setuptools_args = {}

    main(**extra_setuptools_args)
