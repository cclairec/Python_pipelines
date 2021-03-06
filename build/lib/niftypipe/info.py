""" This file contains defines parameters for nipy that we use to fill
settings in setup.py, the nipy top-level docstring, and for building the
docs.  In setup.py in particular, we exec this file, so it cannot import nipy
"""


# nipy version information.  An empty _version_extra corresponds to a
# full release.  '.dev' as a _version_extra string means this is a development
# version
_version_major = 1
_version_minor = 1
_version_micro = 1
_version_extra = '-dev'  # Remove -dev for release


def get_niftypipe_gitversion():
    """niftypipe version as reported by the last commit in git

    Returns
    -------
    None or str
      Version of niftypipe according to git.
    """
    import os
    import subprocess
    try:
        import niftypipe
        gitpath = os.path.realpath(os.path.join(os.path.dirname(niftypipe.__file__),
                                                os.path.pardir))
    except:
        gitpath = os.getcwd()
    gitpathgit = os.path.join(gitpath, '.git')
    if not os.path.exists(gitpathgit):
        return None
    ver = None
    try:
        o, _ = subprocess.Popen('git describe', shell=True, cwd=gitpath,
                                stdout=subprocess.PIPE).communicate()
    except Exception:
        pass
    else:
        ver = o.decode().strip().split('-')[-1]
    return ver

if '-dev' in _version_extra:
    gitversion = get_niftypipe_gitversion()
    if gitversion:
        _version_extra = '-' + gitversion + '.dev'

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
__version__ = "%s.%s.%s%s" % (_version_major,
                              _version_minor,
                              _version_micro,
                              _version_extra)

CLASSIFIERS = ["Development Status :: 5 - Production/Stable",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: MacOS :: MacOS X",
               "Operating System :: POSIX :: Linux",
               "Programming Language :: Python :: 2.7",
               "Programming Language :: Python :: 3.4",
               "Topic :: Scientific/Engineering"]

description = 'Neuroimaging in Python: Pipelines and Interfaces'

# Note: this long_description is actually a copy/paste from the top-level
# README.txt, so that it shows up nicely on PyPI.  So please remember to edit
# it only in one place and sync it correctly.
long_description = \
    """
========================================================
niftypipe: Neuroimaging in Python: Pipelines and Interfaces
========================================================

Current neuroimaging software offer users an incredible opportunity to
analyze data using a variety of different algorithms. However, this has
resulted in a heterogeneous collection of specialized applications
without transparent interoperability or a uniform operating interface.

*niftypipe*, an open-source, community-developed initiative under the
umbrella of NiPy_, is a Python project that provides a uniform interface
to existing neuroimaging software and facilitates interaction between
these packages within a single workflow. niftypipe provides an environment
that encourages interactive exploration of algorithms from different
packages (e.g., ANTS, SPM, FSL, FreeSurfer, Camino, MRtrix, MNE, AFNI, BRAINS,
Slicer), eases the design of workflows within and between packages, and
reduces the learning curve necessary to use different packages. niftypipe is
creating a collaborative platform for neuroimaging software development
in a high-level language and addressing limitations of existing pipeline
systems.

*niftypipe* allows you to:

* easily interact with tools from different software packages
* combine processing steps from different software packages
* develop new workflows faster by reusing common steps from old ones
* process data faster by running it in parallel on many cores/machines
* make your research easily reproducible
* share your processing workflows with the community
"""

# versions
MATPLOTLIB_MIN_VERSION = '1.4.3'
DIPY_MIN_VERSION = '0.7.1'
NIPYPE_MIN_VERSION = '0.12.1'

NAME = 'niftypipe'
MAINTAINER = "niftypipe developers"
MAINTAINER_EMAIL = "m.modat@ucl.ac.uk"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyPipe"
DOWNLOAD_URL = "https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyPipe"
LICENSE = "BSD license"
CLASSIFIERS = CLASSIFIERS
AUTHOR = MAINTAINER
AUTHOR_EMAIL = MAINTAINER_EMAIL
PLATFORMS = ["MacOs",
             "Linux"]
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
ISRELEASE = _version_extra == ''
VERSION = __version__
PROVIDES = ['niftypipe']
REQUIRES = ["nipype>=%s" % NIPYPE_MIN_VERSION,
            "matplotlib>=%s" % MATPLOTLIB_MIN_VERSION,
            "dipy>=%s" % DIPY_MIN_VERSION]
STATUS = 'stable'
