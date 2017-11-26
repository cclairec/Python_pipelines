.. list-table::

 * -  .. image:: images/niftypipe_architecture_overview2.png
         :width: 100 %
   -  .. container::

         **NiftyPipe: workflow management for large scale neuro image algorithms developed at UCL CMIC**

         NiftyPipe is a stand alone python module dedicated to provide a workflow
         environment for algorithms and high level pipelines from `TIG`__ and `DRC`__

         __ tig_
         __ drc_


         **Requirements**

         The NiftyPipe module requires *python>=2.7* installed, and the following list of python modules::

             * numpy>=1.6.2
             * scipy>=0.11
             * networkx>=1.7
             * traits>=4.3
             * python-dateutil>=1.5
             * nibabel>=2.0.1
             * nose>=1.2
             * future==0.15.2
             * simplejson
             * matplotlib==1.4.3
             * dipy>=0.7.1
             * nipype>=0.12.0


         **Installation:**

         For easy installation you can use the following installation line:

         .. code-block:: python

            git clone git@cmiclab.cs.ucl.ac.uk:CMIC/NiftyPipe.git;
            cd NiftyPipe;
            python setup.py install

         NiftyPipe is strongly inspired and based on the `nipype`__ python module.

         __ nipype_

         For internal reasons, we *forked* the module in order to add more interfaces.
         You can find the *forked* repository `here`__.

         __ nipypefork_


.. container::

   **Quick start: usage tof the module:**

   The module is stand alone and can be imported using:

   .. code-block:: python

      import niftypipe

   The module contains `interfaces`__ for different in-house algorithms and commands.
   They are all embedded in a single importable interface called `niftk`. An example of import is:

   __ interfaces_

   .. code-block:: python

      import niftypipe
      from niftypipe.interfaces.niftk import dtitk
      radialdiffusivity = dtitk.TVtool(operation='rd')
      radialdiffusivity.inputs.in_file = <mytensorfieldfile>
      radialdiffusivity.run()


   Here is a non-exhaustive list of interfaces themes provided in the module,
   accessed via **niftypipe.interfaces.niftk**:

   ============ =====
   subinterface Description
   ============ =====
   distorsion   affine distorsion matrix generation
                Gradient non-linearity correction (see `project`__)
   dtitk        diffusion modelling software (see `dtitk webpage`__)
   filters      various common useful filters
   fmri         resting state functional imaging (using `afni`__)
   io           Input Output
   nodditoolbox
   qc
   stats
   utils
   ============ =====

   __ gradwarp_
   __ dtitk_
   __ AFNI_


.. admonition:: Reference

   Gorgolewski K, Burns CD, Madison C, Clark D, Halchenko YO, Waskom ML, Ghosh SS.
   (2011). Nipype: a flexible, lightweight and extensible neuroimaging data
   processing framework in Python. Front. Neuroinform. 5:13. `Download`__

   __ paper_

.. tip::

   To get started, click Quickstart above. The Links box on the right is
   available on any page of this website.

.. include:: links_names.txt
