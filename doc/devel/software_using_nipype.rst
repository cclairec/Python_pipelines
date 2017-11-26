.. _software_using_niftypipe:

=====================
Software using NiftyPipe
=====================

Configurable Pipeline for the Analysis of Connectomes (C-PAC)
-------------------------------------------------------------

`C-PAC <http://fcp-indi.github.io/>`_ is an open-source software pipeline for automated preprocessing and analysis of resting-state fMRI data. C-PAC builds upon a robust set of existing software packages including AFNI, FSL, and ANTS, and makes it easy for both novice users and experts to explore their data using a wide array of analytic tools. Users define analysis pipelines by specifying a combination of preprocessing options and analyses to be run on an arbitrary number of subjects. Results can then be compared across groups using the integrated group statistics feature. C-PAC makes extensive use of NiftyPipe Workflows and Interfaces.

BRAINSTools
-----------
`BRAINSTools <http://brainsia.github.io/BRAINSTools/>`_ is a suite of tools for medical image processing focused on brain analysis.

Brain Imaging Pipelines (BIPs)
------------------------------

`BIPs <https://github.com/INCF/BrainImagingPipelines>`_ is a set of predefined NiftyPipe workflows coupled with a graphical interface and ability to save and share workflow configurations. It provides both NiftyPipe Workflows and Interfaces.

BROCCOLI
--------

`BROCCOLI <https://github.com/wanderine/BROCCOLI/>`_ is a piece of software for fast fMRI analysis on many core CPUs and GPUs. It provides NiftyPipe Interfaces.

Forward
-------

`Forward <http://cyclotronresearchcentre.github.io/forward/>`_ is set of tools simplifying the preparation of accurate electromagnetic head models for EEG forward modeling. It uses NiftyPipe Workflows and Interfaces.

Limbo
-----

`Limbo <https://github.com/Gilles86/in_limbo>`_ is a toolbox for finding brain regions that are neither significantly active nor inactive, but rather “in limbo”. It was build using custom NiftyPipe Interfaces and Workflows.

Lyman
-----

`Lyman <http://stanford.edu/~mwaskom/software/lyman/>`_ is a high-level ecosystem for analyzing task based fMRI neuroimaging data using open-source software. It aims to support an analysis workflow that is powerful, flexible, and reproducible, while automating as much of the processing as possible. It is build upon NiftyPipe Workflows and Interfaces.

Medimsight
----------

`Medimsight <https://www.medimsight.com>`_ is a commercial service medical imaging cloud platform. It uses NiftyPipe to interface with various neuroimaging software.

MIA
---

`MIA <http://mia.sourceforge.net>`_ MIA is a a toolkit for gray scale medical image analysis. It provides NiftyPipe interfaces for easy integration with other software.

Mindboggle
----------

`Mindboggle <http://mindboggle.info/users/README.html>`_ software package automates shape analysis of anatomical labels and features extracted from human brain MR image data. Mindboggle can be run as a single command, and can be easily installed as a cross-platform virtual machine for convenience and reproducibility of results. Behind the scenes, open source Python and C++ code run within a NiftyPipe pipeline framework.

OpenfMRI
--------

`OpenfMRI <https://openfmri.org/>`_ is a repository for task based fMRI datasets. It uses NiftyPipe for automated analysis of the deposited data.

serial functional Diffusion Mapping (sfDM)
------------------------------------------

'sfDM <http://github.com/PIRCImagingTools/sfDM>'_ is a software package for looking at changes in diffusion profiles of different tissue types across time. It uses NiftyPipe to process the data.


The Stanford CNI MRS Library (SMAL)
-----------------------------------

`SMAL <http://cni.github.io/MRS/doc/_build/html/index.html>`_ is a library providing algorithms and methods to read and analyze data from Magnetic Resonance Spectroscopy (MRS) experiments. It provides an API for fitting models of the spectral line-widths of several different molecular species, and quantify their relative abundance in human brain tissue. SMAL uses NiftyPipe Workflows and Interfaces.

tract_querier
-------------

`tract_querier <https://github.com/demianw/tract_querier>`_ is a White Matter Query Language tool. It provides NiftyPipe interfaces.
