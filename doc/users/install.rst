.. _install:

======================
 Download and install
======================

This page covers the necessary steps to install NiftyPipe.

Download
--------

.. Development: [`zip <https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyPipe/zipball/master>`__ `tar.gz
.. <https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyPipe/tarball/master>`__]

.. `Prior downloads <http://github.com/nipy/niftypipe/tags>`_

To check out the latest development version::

        git clone git@cmiclab.cs.ucl.ac.uk:CMIC/NiftyPipe.git

Install
-------

The installation process is similar to other Python packages.

If you already have a Python environment setup that has the dependencies listed
below, you can do::

	easy_install niftypipe

or::

	pip install niftypipe

Debian and Ubuntu
~~~~~~~~~~~~~~~~~

Add the `NeuroDebian <http://neuro.debian.org>`_ repository and install
the ``python-niftypipe`` package using ``apt-get`` or your favorite package
manager.

Mac OS X
~~~~~~~~

The easiest way to get niftypipe running on Mac OS X is to install Anaconda_ or
Canopy_ and then add nibabel and niftypipe by executing::

	easy_install nibabel
	easy_install niftypipe

From source
~~~~~~~~~~~

If you downloaded the source distribution named something
like ``niftypipe-x.y.tar.gz``, then unpack the tarball, change into the
``niftypipe-x.y`` directory and install niftypipe using::

    python setup.py install

**Note:** Depending on permissions you may need to use ``sudo``.

Testing the install
-------------------

The best way to test the install is to run the test suite.  If you have
nose_ installed, then do the following::

    python -c "import niftypipe; niftypipe.test()"

you can also test with nosetests::

    nosetests --with-doctest /software/nipy-repo/masterniftypipe/niftypipe
    --exclude=external --exclude=testing

All tests should pass (unless you're missing a dependency). If SUBJECTS_DIR
variable is not set some FreeSurfer related tests will fail. If any tests
fail, please report them on our `bug tracker
<https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyPipe/issues>`_.

On Debian systems, set the following environment variable before running
tests::

       export MATLABCMD=$pathtomatlabdir/bin/$platform/MATLAB

where, $pathtomatlabdir is the path to your matlab installation and
$platform is the directory referring to x86 or x64 installations
(typically glnxa64 on 64-bit installations).

Avoiding any MATLAB calls from testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On unix systems, set an empty environment variable::

    export NIPYPE_NO_MATLAB=

This will skip any tests that require matlab.

Dependencies
------------

Below is a list of required dependencies, along with additional software
recommendations.

Must Have
~~~~~~~~~

Python_ 2.7

Nibabel_ 1.0 - 1.4
  Neuroimaging file i/o library

NetworkX_ 1.0 - 1.8
  Python package for working with complex networks.

NumPy_ 1.3 - 1.7

SciPy_ 0.7 - 0.12
  Numpy and Scipy are high-level, optimized scientific computing libraries.

Enthought_ Traits_ 4.0.0 - 4.3.0

Dateutil 1.5 -

.. note::

    Full distributions such as Anaconda_ or Canopy_ provide the above packages,
    except Nibabel_.

Strong Recommendations
~~~~~~~~~~~~~~~~~~~~~~

IPython_ 0.10.2 - 1.0.0
  Interactive python environment. This is necessary for some parallel
  components of the pipeline engine.

Matplotlib_ 1.0 - 1.4.3
  Plotting library

`RDFLib <http://rdflib.readthedocs.org/en/latest/>`_ 4.1
RDFLibrary required for provenance export as RDF

Sphinx_ 1.1
  Required for building the documentation

`Graphviz <http://www.graphviz.org/>`_
  Required for building the documentation

Interface Dependencies
~~~~~~~~~~~~~~~~~~~~~~

These are the software packages that niftypipe.interfaces wraps:

FSL_
  4.1.0 or later

matlab_
  2008a or later

SPM_
  SPM5/8

FreeSurfer_
  FreeSurfer version 4 and higher

AFNI_
  2009_12_31_1431 or later

Slicer_
  3.6 or later

Nipy_
  0.1.2+20110404 or later

Nitime_
  (optional)

Camino_

Camino2Trackvis_

ConnectomeViewer_

.. include:: ../links_names.txt
