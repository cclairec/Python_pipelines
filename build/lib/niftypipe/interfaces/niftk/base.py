# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
import warnings
from exceptions import NotImplementedError
from distutils import spawn
from nipype.interfaces.base import (traits, CommandLineInputSpec)
from nipype.utils.misc import str2bool
from nipype import config
import shutil

warn = warnings.warn
warnings.filterwarnings('always', category=UserWarning)


class Info(object):
    """Handle niftk output type and version information.

    version refers to the version of niftk on the system

    output type refers to the type of file niftk defaults to writing
    eg, NIFTI, NIFTI_GZ

    """

    ftypes = {'NIFTI': '.nii',
              'NIFTI_PAIR': '.img',
              'NIFTI_GZ': '.nii.gz',
              'NIFTI_PAIR_GZ': '.img.gz'}

    @staticmethod
    def version():
        """Check for niftk version on system

        Parameters
        ----------
        None

        Returns
        -------
        version : str
           Version number as string or None if niftyreg not found

        """
        raise NotImplementedError("Waiting for Niftk version fix before  implementing this")

    @classmethod
    def output_type_to_ext(cls, output_type):
        """Get the file extension for the given output type.

        Parameters
        ----------
        output_type : {'NIFTI', 'NIFTI_GZ', 'NIFTI_PAIR', 'NIFTI_PAIR_GZ'}
            String specifying the output type.

        Returns
        -------
        extension : str
            The file extension for the output type.
        """

        try:
            return cls.ftypes[output_type]
        except KeyError:
            msg = 'Invalid NiftkOutputType: ', output_type
            raise KeyError(msg)


class NIFTKCommandInputSpec(CommandLineInputSpec):
    """
    Base Input Specification for all Niftk Commands

    All command support specifying the output type dynamically
    via output_type.
    """
    output_type = traits.Enum('NIFTI_GZ', Info.ftypes.keys(),
                              desc='Niftk output type')


def no_niftk():
    """Checks if niftk is NOT installed
    """
    raise NotImplementedError("Waiting for version fix")


# A common function to create the workflow graph
def generate_graph(workflow):
    """
    A common function to create the workflow graph

    :param workflow: The workflow to generate the graph from

    :return:
    """
    dot_exec = spawn.find_executable('dot')
    if dot_exec is not None:
        workflow.write_graph(graph2use='colored')


# Default arguments for the perform_* parser
def default_parser_argument(parser):
    """
    Default arguments for the perform_* parser

    :param parser: The parser to add the extra default arguments to

    :return: None
    """
    parser.add_argument('--no_qsub',
                        dest='use_qsub',
                        help='[QSUB] Use this option to forbid internal qsub submission',
                        action='store_false',
                        default=True,
                        required=False)
    parser.add_argument('--openmp_core',
                        dest='openmp_core',
                        metavar='INT',
                        type=int,
                        help='Number of openmp cores to use for niftyreg and niftyseg',
                        default=1,
                        required=False)
    parser.add_argument('-n', '--n_procs',
                        type=int,
                        dest='n_procs',
                        metavar='INT',
                        help='maximum number of CPUs to be used when using the MultiProc plugin. ',
                        required=False,
                        default=-1)
    parser.add_argument('-u', '--username',
                        dest='username',
                        metavar='NAME',
                        help='[QSUB] Username to use to submit jobs on the cluster',
                        required=False)
    parser.add_argument('-g', '--graph',
                        dest='graph',
                        help='Print a graph describing the node connections',
                        action='store_true',
                        default=False,
                        required=False)
    parser.add_argument('--remove_tmp',
                        dest='remove_tmp',
                        help='Remove/Delete the directory containing all nodes files after execution. Default is false',
                        action='store_true',
                        default=False,
                        required=False)
    parser.add_argument('--working_dir',
                        dest='working_dir',
                        metavar='DIR',
                        help='Control the base_dir of the workflow where temporary files are accumulated (default is output_dir)',
                        required=False)


# A common function to use by all pipelines to run workflow
def run_workflow(workflow, parser=None, qsubargs=None):
    """

    A common function to use by all pipelines to run workflow

    :param workflow: workflow to run
    :param parser: the parser from which to take the ret of the arguments:
        n_proc: the (maximum) amount of processors to be used in the MultiProc mode
        username: the username to use as login in the cluster, to avoid the workflow to stop when not connected
        use_qsub: boolean to force *not* to use the cluster mode
        openmp_core: the number of cores to parse in the environment for OPENMP, used in niftyreg and niftyseg
        remove_tmp: Remove/Delete the directory containing all nodes files after execution.
        Default is false
    :param qsubargs: argument to parse to the cluster in the qsub

    :return: None
    """

    """
    Force the datasink to copy files rather than creating simlink when using a renamer
    """
    config.update_config({'execution': {'try_hard_link_datasink': 'false'}})

    """
    Modify the base_dir of the workflow in case the user has required it
    """
    if parser and parser.working_dir:
        if not os.path.exists(os.path.abspath(parser.working_dir)):
            os.mkdir(os.path.abspath(parser.working_dir))
        workflow.base_dir = os.path.abspath(parser.working_dir)

    """
    If provided by the user Set the number of openmp core to use
    """

    if parser and parser.openmp_core > 0:
        os.environ['OMP_NUM_THREADS'] = str(parser.openmp_core)

    """
    We do not want a report creation in the working directory as we do not know if we have right access to it
    """

    config.update_config({'execution': {'create_report': 'false'}})

    """
    By default we estimate the user wants to use the cluster capabilities if available
    """

    run_qsub = True

    """
    If it has been set by the user, grab env variable "RUN_QSUB" and parse it
    """

    if os.getenv('RUN_QSUB'):
        run_qsub = str2bool(os.getenv('RUN_QSUB'))

    """
    Only go through the cluster if all requirements are fullfilled
    """

    if run_qsub and parser and parser.use_qsub and spawn.find_executable('qsub'):

        """
        Default qsub arguments to parse
        """

        qargs = '-l h_rt=00:05:00 -l tmem=1.8G -l h_vmem=1.8G -l vf=1.8G ' + \
                '-l s_stack=10240 -j y -b y -S /bin/csh -V'

        """
        If provided in the arguments or in the env variable, grab the qsub arguments
        """

        if qsubargs:
            qargs = qsubargs
        elif os.getenv('QSUB_OPTIONS'):
            qargs = os.getenv('QSUB_OPTIONS')

        """
        Parse the openmp number of cores into the qsub arguments
        """

        if parser and parser.openmp_core > 1:
            qargs = qargs + ' -pe smp ' + str(parser.openmp_core)

        """
        The plugin arguments include the qsubargs
        """

        pargs = {'qsub_args': qargs}
        if parser and parser.username:
            pargs = {'qsub_args': qsubargs, 'username': parser.username}

        """
        Run the workflow using the SGE plugin
        """

        workflow.run(plugin='SGE', plugin_args=pargs)

    elif parser:

        """
        If there is a 'parser' in the arguments, grab the arguments from the command line and parse them
        to see which plugin to use and how many procs in case of 'MultiProc'
        """

        plugin = 'MultiProc'  # By default, we assume the user wants to use all procs in the machine: 'MultiProc'
        pargs = {}
        if parser.n_procs == 1:  # Only if the user requests a single proc to use, we use the 'Linear' plugin
            plugin = 'Linear'
        elif parser.n_procs > 1:  # If the user requires a specific amount of procs we parse this to the plugin args
            pargs = {'n_procs': parser.n_procs}
        workflow.run(plugin=plugin, plugin_args=pargs)

    else:

        """
        If no 'parser' was used in the arguments we run the workflow with default behaviour (or the one in .nipype.cfg)
        """

        workflow.run()

    """
    After the successful run of the workflow (if unsuccessful it raises an error and exits),
    we attempt to remove the temporary directories created by the run.
    """
    to_remove = ''
    if workflow.base_dir and os.path.exists(workflow.base_dir):
        to_remove = os.path.abspath(os.path.join(workflow.base_dir, workflow.name))
    if parser and parser.remove_tmp and workflow.base_dir and os.path.exists(to_remove):
        print 'removing the temporary directory: %s' % to_remove
        shutil.rmtree(to_remove)
        print 'done.'

    return
