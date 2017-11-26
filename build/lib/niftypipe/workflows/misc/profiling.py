# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe


def data_to_1D_functions_function(in_data_image, axis_of_symmetry=0):
    import numpy as np
    import nibabel as nib
    import os

    data_image = nib.load(in_data_image)
    qform = data_image.get_qform()
    indices = np.where(data_image.get_data() > 0)
    points = np.array(
        [nib.affines.apply_affine(qform, [i, j, k]) for i, j, k in zip(indices[0], indices[1], indices[2])]).transpose()
    data = np.array([data_image.get_data()[i, j, k] for i, j, k in zip(indices[0], indices[1], indices[2])]).transpose()
    xmean = np.mean(points[axis_of_symmetry])
    structure_1 = points[:, points[axis_of_symmetry, :] < xmean]
    structure_2 = points[:, points[axis_of_symmetry, :] > xmean]
    data_1 = data[points[axis_of_symmetry, :] < xmean]
    data_2 = data[points[axis_of_symmetry, :] > xmean]

    mean_1 = np.mean(structure_1, axis=1)[:, np.newaxis]
    mean_2 = np.mean(structure_2, axis=1)[:, np.newaxis]

    differences = structure_1 - mean_1
    m_1 = np.zeros([3, 3])
    m_1[0][0] = np.mean(differences[0, :] * differences[0, :])
    m_1[0][1] = m_1[1][0] = np.mean(differences[0, :] * differences[1, :])
    m_1[0][2] = m_1[2][0] = np.mean(differences[0, :] * differences[2, :])
    m_1[1][1] = np.mean(differences[1, :] * differences[1, :])
    m_1[1][2] = m_1[2][1] = np.mean(differences[1, :] * differences[2, :])
    m_1[2][2] = np.mean(differences[2, :] * differences[2, :])

    differences = structure_2 - mean_2
    m_2 = np.zeros([3, 3])
    m_2[0][0] = np.mean(differences[0, :] * differences[0, :])
    m_2[0][1] = m_2[1][0] = np.mean(differences[0, :] * differences[1, :])
    m_2[0][2] = m_2[2][0] = np.mean(differences[0, :] * differences[2, :])
    m_2[1][1] = np.mean(differences[1, :] * differences[1, :])
    m_2[1][2] = m_2[2][1] = np.mean(differences[1, :] * differences[2, :])
    m_2[2][2] = np.mean(differences[2, :] * differences[2, :])

    U1, _, _ = np.linalg.svd(m_1)
    U2, _, _ = np.linalg.svd(m_2)
    axis_1 = U1[:, 0][:, np.newaxis]
    axis_2 = U2[:, 0][:, np.newaxis]
    if axis_1[1] > 0:
        axis_1 = - axis_1
    if axis_2[1] > 0:
        axis_2 = - axis_2

    absissa_1 = np.dot(structure_1.transpose(), axis_1).reshape(structure_1.shape[1])
    absissa_2 = np.dot(structure_2.transpose(), axis_2).reshape(structure_2.shape[1])

    function_1 = np.array([absissa_1, data_1]).transpose()
    function_2 = np.array([absissa_2, data_2]).transpose()

    np.savetxt('structure_1.txt', function_1, fmt='%5.10f')
    np.savetxt('structure_2.txt', function_2, fmt='%5.10f')
    np.savetxt('axis_1.txt', axis_1, fmt='%5.10f')
    np.savetxt('axis_2.txt', axis_2, fmt='%5.10f')

    return os.path.abspath('structure_1.txt'), os.path.abspath('structure_2.txt'), \
           os.path.abspath('axis_1.txt'), os.path.abspath('axis_2.txt')


def convolve_integrate_function(in_function_file, profilesize):
    import numpy as np
    from scipy import stats
    import os

    in_function = np.loadtxt(in_function_file)
    bins = np.linspace(min(in_function[:, 0]), max(in_function[:, 0]), profilesize)
    total_function = sum(in_function[:, 1])
    kernel = stats.gaussian_kde(in_function[:, 0], 'silverman')
    data = total_function * kernel(bins) / (sum(kernel(bins)) / len(bins))
    np.savetxt('function_profile.txt', np.array([bins, data]).transpose(), fmt='%5.5f')
    return os.path.abspath('function_profile.txt')


def graph_generator_function(in_profile_file_1, in_profile_file_2, xlabel, ylabel_1, ylabel_2):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    in_profile_1 = np.loadtxt(in_profile_file_1)
    in_profile_2 = np.loadtxt(in_profile_file_2)
    numberofbins = in_profile_1.shape[0]
    mean_profile_1 = np.mean(in_profile_1[:, 1])
    mean_profile_2 = np.mean(in_profile_2[:, 1])
    ymax = np.max(np.append(in_profile_1[:, 1], in_profile_2[:, 1]))
    fig = plt.figure()
    ax = plt.subplot(111)
    x_axis = np.linspace(0, 1, numberofbins)
    plt.plot(x_axis, in_profile_1[:, 1], 'r-', label=ylabel_1)
    plt.plot(x_axis, in_profile_2[:, 1], 'g-', label=ylabel_2)
    plt.plot(x_axis, mean_profile_1 * np.ones(numberofbins), 'r--')
    plt.plot(x_axis, mean_profile_2 * np.ones(numberofbins), 'g--')
    plt.text(0.75, 0.75,
             'total: %2.3f %%' % float(100 * mean_profile_1),
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes,
             color='red')
    plt.text(0.75, 0.70,
             'total: %2.3f %%' % float(100 * mean_profile_2),
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes,
             color='green')
    plt.xlabel(xlabel)
    plt.legend(loc='best', fontsize='small')
    plt.ylim([0, ymax])
    plt.grid(which='major', axis='both')
    plt.tight_layout()
    fig.savefig('graph.png', format='PNG')
    plt.close()

    return os.path.abspath('graph.png')


def create_dual_structure_1D_profile_pipeline(name='dual_structure_1D_profile_workflow'):
    """
    """

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    input_node = pe.Node(
        interface=niu.IdentityInterface(
            fields=['in_dual_structure_image',
                    'in_profilesize',
                    'in_axis_of_symmetry',
                    'in_xlabel',
                    'in_ylabel_1',
                    'in_ylabel_2']),
        name='input_node')

    '''
    *****************************************************************************
    First step: Create 1D function of axis of mass abscissa
    *****************************************************************************
    '''

    image_to_x_functions = pe.Node(niu.Function(input_names=['in_data_image', 'axis_of_symmetry'],
                                                output_names=['out_function_1', 'out_function_2',
                                                              'out_axis_1', 'out_axis_2'],
                                                function=data_to_1D_functions_function),
                                   name='image_to_x_functions')
    workflow.connect(input_node, 'in_axis_of_symmetry', image_to_x_functions, 'axis_of_symmetry')
    workflow.connect(input_node, 'in_dual_structure_image', image_to_x_functions, 'in_data_image')

    '''
    *****************************************************************************
    Second step: Convolve/Integrate the 1D functions with Gaussian kernel at bin points
    *****************************************************************************
    '''

    profile_generator_1 = pe.Node(niu.Function(input_names=['in_function_file', 'profilesize'],
                                               output_names=['out_profile'],
                                               function=convolve_integrate_function),
                                  name='profile_generator_1')
    profile_generator_2 = pe.Node(niu.Function(input_names=['in_function_file', 'profilesize'],
                                               output_names=['out_profile'],
                                               function=convolve_integrate_function),
                                  name='profile_generator_2')

    workflow.connect(input_node, 'in_profilesize', profile_generator_1, 'profilesize')
    workflow.connect(input_node, 'in_profilesize', profile_generator_2, 'profilesize')
    workflow.connect(image_to_x_functions, 'out_function_1', profile_generator_1, 'in_function_file')
    workflow.connect(image_to_x_functions, 'out_function_2', profile_generator_2, 'in_function_file')

    '''
    *****************************************************************************
    Third step: Generate a graph with the data profile
    *****************************************************************************
    '''

    graph_generator = pe.Node(
        niu.Function(input_names=['in_profile_file_1', 'in_profile_file_2', 'xlabel', 'ylabel_1', 'ylabel_2'],
                     output_names=['out_graph'],
                     function=graph_generator_function),
        name='graph_generator')
    workflow.connect(input_node, 'in_xlabel', graph_generator, 'xlabel')
    workflow.connect(input_node, 'in_ylabel_1', graph_generator, 'ylabel_1')
    workflow.connect(input_node, 'in_ylabel_2', graph_generator, 'ylabel_2')
    workflow.connect(profile_generator_1, 'out_profile', graph_generator, 'in_profile_file_1')
    workflow.connect(profile_generator_2, 'out_profile', graph_generator, 'in_profile_file_2')

    '''
    *****************************************************************************
    Fourth step: rename profiles for convenience
    *****************************************************************************
    '''

    profile_renamer_1 = pe.Node(niu.Rename(format_string='profile_1.txt'), name='profile_renamer_1')
    workflow.connect(profile_generator_1, 'out_profile', profile_renamer_1, 'in_file')
    profile_renamer_2 = pe.Node(niu.Rename(format_string='profile_2.txt'), name='profile_renamer_2')
    workflow.connect(profile_generator_2, 'out_profile', profile_renamer_2, 'in_file')

    '''
    *****************************************************************************
    Connect the outputs to the output node
    *****************************************************************************
    '''

    output_node = pe.Node(interface=niu.IdentityInterface(
                          fields=['out_profile_1', 'out_profile_2', 'out_graph']),
                          name='output_node')

    workflow.connect(profile_renamer_1, 'out_file', output_node, '@out_profile_1')
    workflow.connect(profile_renamer_2, 'out_file', output_node, '@out_profile_2')
    workflow.connect(image_to_x_functions, 'out_axis_1', output_node, '@out_axis_1')
    workflow.connect(image_to_x_functions, 'out_axis_2', output_node, '@out_axis_2')
    workflow.connect(graph_generator, 'out_graph', output_node, '@out_graph')

    return workflow
