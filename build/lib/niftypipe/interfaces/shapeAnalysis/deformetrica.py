# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from nipype.interfaces.base import (CommandLineInputSpec, BaseInterfaceInputSpec,BaseInterface,
                                    TraitedSpec, File, traits, InputMultiPath, isdefined, OutputMultiPath)
from nipype.utils.filemanip import split_filename
import os

from .base import DeformetricaCommand


class ParallelTransportInputSpec(CommandLineInputSpec):
    in_cp = File(argstr="-cp %s", exists=True, mandatory=True,
                 desc='txt file containing the coordinates of the initial control points ' +
                      'needed to parametrise the geodesic (madatory)')

    in_paramDiffeo = File(argstr="-d %s", exists=True, mandatory=True,
                          desc="xml file with the parameters for the Diffeos (mandatory)")

    in_paramObjects = File(argstr="-op %s", exists=True, mandatory=True,
                           desc="the list of xml files with the parameters for the shapes (mandatory) ")

    in_mom = File(argstr="-mom %s", exists=True, mandatory=True,
                  desc='txt file containing the initial momentum vectors defined on the CP, ' +
                       'needed to parametrise the geodesic (madatory)')

    in_vect = File(argstr="-v %s", exists=True, mandatory=True,
                   desc='txt file containing the vector we want to transport. ' +
                        'Same organisation as for a momentum vector file.')

    in_boolMom = traits.Int(argstr="-isMom %s", mandatory=False,
                            desc='default = 1. indicates if the vector to transport is ' +
                                  'already a velocity vector field (false or 0), or a momentum vector ' +
                                  '(true or 1). If a momentum vector is given, the corresponding velocity ' +
                                  'vector field will be computed, and a transported momentum vector will be returned')

    in_t_from = traits.Int(argstr="-from %s", mandatory=True,
                           desc="the index of the time point where the vector to transport is defined ")

    in_t_to = traits.Int(argstr="-to %s", mandatory=True,
                         desc="the index of the time point where we want to transport the vector ")

    in_vtk = InputMultiPath(argstr="-f %s", exists=True, mandatory=True,
                  desc="vtk file(s) (one file per object) of the shape in which the initial conditions are determined ")
    out_file = traits.String(argstr="-o %s",
                             name_source=['in_vect'],
                             name_template='%s_TRANSPORT',
                             desc="template for the output file names: " +
                                  "transported vector files will be <templateFilename>_t_i.txt " +
                                  "for i between time from and time to.",
                             keep_extension=False)
    desc = "Parallel transport of the vector (-v) along the geodesic parametrised by -cp and -mom"
    out_transported_vect = traits.String(name_source=['out_file'], name_template='%s_0.txt', desc=desc)
    desc = "Parallel transport of the vector (-v) along the geodesic parametrised by -cp and -mom"
    out_transported_mom = traits.String(name_source=['out_file'], name_template='%s_Mom_0.txt', desc=desc)


class ParallelTransportOutputSpec(TraitedSpec):
    out_file = File(desc=' CLAIRE!')
    out_transported_vect = File(desc=' CLAIRE!')
    out_transported_mom = File(desc=' CLAIRE!')


class ParallelTransport(DeformetricaCommand):
    _cmd = "parallelTransport"
    input_spec = ParallelTransportInputSpec
    output_spec = ParallelTransportOutputSpec

    # def _format_arg(self, opt, spec, val):
    #     """Convert input to appropriate format for ParallelTransport."""
    #     if opt == 'in_bool_mom':
    #         return '-isMom %d' % 1 if self.inputs.in_bool_mom else 0
    #     else:
    #         return super(ParallelTransport, self)._format_arg(opt, spec, val)

    # def _list_outputs(self):
    #     outputs = self.output_spec().get()
    #     print outputs['out_file']
    #     outputs['out_transported_vect'] = os.path.abspath(str(outputs['out_file']) + '_0.txt')
    #     print outputs['out_transported_vect']
    #     outputs['out_transported_mom'] = os.path.abspath('Mom0_final_TRANSPORT_Mom_0.txt')
    #     return outputs


class SparseGeodesicRegression3InputSpec(CommandLineInputSpec):
    in_paramDiffeo = File(argstr="%s", exists=True, mandatory=True, position=0,
                          desc='xml file containing the parameters for the diffeo computation ')

    in_nbOfObjects = traits.Int(1, argstr="%s", usedefault=True, position=1,
                                desc="Number of objects (not subjects)")

    in_paramObjects = InputMultiPath(argstr="%s", exists=True, mandatory=True, position=-1,
                                     desc='xml files containing the parameters of objects ')

    in_initTemplates = InputMultiPath(exists=True, mandatory=True,
                                      desc='vtk files containing the vtk mesh of the initial template for objects')

    in_subjects = InputMultiPath(exists=True, mandatory=True,
                                 desc='vtk files containing the vtk mesh of the object of the subject ')
    in_time = File(exists=True, mandatory=True,
                   desc='ages or times corresponding to the subjects ')


class SparseGeodesicRegression3OutputSpec(TraitedSpec):
    out_file_CP = File(desc="Control points where the momenta of the geodesic regression are defined")
    out_file_MOM = File(desc="Initial momenta of the geodesic regression")
    out_vtk_file_trajectories = traits.List(File(desc="list of vtk files, trajectory (one file per time point) " +
                                                      "computed at each iteration"))
    out_log = File("out.log", desc="log file of the method. Can be needed to check how the method ran.")


class SparseGeodesicRegression3(DeformetricaCommand):
    _cmd = "sparseGeodesicRegression3"
    input_spec = SparseGeodesicRegression3InputSpec
    output_spec = SparseGeodesicRegression3OutputSpec


    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file_CP'] = os.path.abspath("CP_final.txt")
        outputs['out_file_MOM'] = os.path.abspath("Mom0_final.txt")
        return outputs

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for sparseGeodesicRegression3."""
        if opt == 'in_paramObjects':
            # create a list of EYO normalised from the txt file generated by the interface compute
            ages = []
            with open(self.inputs.in_time, "r") as file_ages:
                for x in file_ages:
                    ages.append(float(x.split()[2]))

            # Write the input as a list of trait.Files
            arg = ''
            if len(self.inputs.in_paramObjects) != self.inputs.in_nbOfObjects or \
                    len(self.inputs.in_initTemplates) != self.inputs.in_nbOfObjects:
                # error
                msg = "Number of in_paramObjects(%d) different than the number of objects (%d)"
                raise Exception(msg % (len(self.inputs.in_paramObjects), self.inputs.in_nbOfObjects))
            else:
                if len(self.inputs.in_subjects) != len(ages):
                    msg = "Number of in_subjects(%d) different than the number of ages (%d)"
                    raise Exception(msg % (len(self.inputs.in_subjects), len(ages)))
                else:
                    for index, paramObj in enumerate(self.inputs.in_paramObjects):
                        arg += '%s %s ' % (paramObj, self.inputs.in_initTemplates[index])
                        for i, vtkMesh in enumerate(self.inputs.in_subjects):
                            arg += '%s %s ' % (vtkMesh, ages[i])

                    arg += ' > out.log'
                    return arg
        else:
            return super(SparseGeodesicRegression3, self)._format_arg(opt, spec, val)


class SparseAtlas3InputSpec(CommandLineInputSpec):
    in_paramDiffeo = File(argstr="%s", exists=True, mandatory=True, position=0,
                          desc='xml file containing the parameters for the diffeo computation ')

    in_nbOfObjects = traits.Int(1, argstr="%s", usedefault=True, position=1,
                                desc="Number of objects (not subjects)")

    in_paramObjects = InputMultiPath(argstr="%s", exists=True, mandatory=True, position=-1,
                                     desc='xml files containing the parameters of objects ')
    
    in_initTemplates = InputMultiPath(exists=True, mandatory=True,
                                      desc='vtk files containing the vtk mesh of the initial template for objects')

    in_subjects = InputMultiPath(exists=True, mandatory=True,
                                 desc='vtk files containing the vtk mesh of the initial template of the objects ')

class SparseAtlas3OutputSpec(TraitedSpec):
    out_file_CP = File(desc="a text file containing the coordinates of the optimal position of the control points")
    out_file_MOM = File(desc="a text file containing the values of the optimal momentum vectors attached to " +
                             "the control points, together parameterising the optimal flow of deformation \phi_t " +
                             "warping the source to the target")
    out_files_vtk = traits.List(File(exists=True, desc="vtk files of the final template. One file per object."))
    out_file_vtk = File(exists=True, desc=" One file per object. When one object. the final template")


class SparseAtlas3(DeformetricaCommand):
    """ This command allows the registration between two sets of shapes. A set of shapes is made of the union of
    different objects (Object1, Object2, etc..) embedded in the ambient 2D or 3D space.
    """
    _cmd = "sparseAtlas3"
    input_spec = SparseAtlas3InputSpec
    output_spec = SparseAtlas3OutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file_MOM'] = os.path.abspath('Mom_final.txt')
        outputs['out_file_CP'] = os.path.abspath('CP_final.txt')
        init_templates = self.inputs.in_initTemplates
        list_outfile = []
        print "================ " + str(len(init_templates)) + "====================="
        if len(init_templates) > 1:
            for init_temp in init_templates:
                p, bn, ext = split_filename(init_temp)
                list_outfile.append(p + '/' + bn + '_template.vtk')
            outputs['out_files_vtk'] = list_outfile
        else:
            p, bn, ext = split_filename(init_templates[0])
            outputs['out_file_vtk'] = p + '/' + bn + '_template.vtk'
        return outputs

    def _format_arg(self, opt, spec, val):
        if opt == 'in_paramObjects':
            # Write the input as a list of trait.Files
            arg = ''
            if len(self.inputs.in_paramObjects) != self.inputs.in_nbOfObjects or \
                    len(self.inputs.in_initTemplates) != self.inputs.in_nbOfObjects:
                # error
                msg = "Number of in_paramObjects(%d) different than the number of objects (%d)"
                raise Exception(msg % (len(self.inputs.in_paramObjects), self.inputs.in_nbOfObjects))
            else:
                for index, paramObj in enumerate(self.inputs.in_paramObjects):
                    arg += '%s %s ' % (paramObj, self.inputs.in_initTemplates[index])
                    for i, vtkMesh in enumerate(self.inputs.in_subjects):
                        arg += '%s ' % vtkMesh

                arg += ' > out_sparseAtlas3.log'
                return arg
        else:
            return super(SparseAtlas3, self)._format_arg(opt, spec, val)



class SparseMatching3InputSpec(CommandLineInputSpec):
    in_paramDiffeo = File(argstr="%s", exists=True, mandatory=True, position=0,
                          desc='xml file containing the parameters for the diffeo computation ')

    in_paramObjects = InputMultiPath(exists=True, mandatory=True,
                                     desc='xml files containing the parameters of objects. ' +
                                          'Works for only one object at the moment (need to work on the workflow). ')

    in_sources = InputMultiPath(exists=True, mandatory=True,
                                desc='vtk files containing the mesh of the source. 1 object, 1 source file.')

    in_targets = InputMultiPath(exists=True, mandatory=True,
                                desc="vtk files containing the mesh of the target. 1 object, 1 target file")


class SparseMatching3OutputSpec(TraitedSpec):
    out_file_CP = File(desc="a text file containing the coordinates of the optimal position of the control points")
    out_file_MOM = File(desc="a text file containing the values of the optimal momentum vectors attached to " +
                             "the control points, together parameterising the optimal flow of deformation \phi_t " +
                             "warping the source to the target")
    #out_files_vtk = OutputMultiPath(File(exists=True), desc="a set of files of the same format as the kth input " +
    #                                                        "object file. These files give the deformation of " +
    #                                                        "the object along the optimal deformation")


class SparseMatching3(DeformetricaCommand):
    """ This command allows the registration between two sets of shapes. A set of shapes is made of the union of
    different objects (Object1, Object2, etc..) embedded in the ambient 2D or 3D space.
    """
    _cmd = "sparseMatching3"
    input_spec = SparseMatching3InputSpec
    output_spec = SparseMatching3OutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file_MOM'] = os.path.abspath('Mom_final.txt')
        outputs['out_file_CP'] = os.path.abspath('CP_final.txt')
        return outputs

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for SparseMatching3."""
        if opt == 'in_paramDiffeo':
            # Write the input as a list of trait.Files
            arg = ' ' + self.inputs.in_paramDiffeo + ' '
            if len(self.inputs.in_paramObjects) != len(self.inputs.in_sources):
                # error
                msg = "Number of in_paramObjects(%d) different than the number of sources (%d)"
                raise Exception(msg % (len(self.inputs.in_paramObjects), len(self.inputs.in_sources)))
            else:
                if len(self.inputs.in_sources) != len(self.inputs.in_targets):
                    msg = "Number of in_subjects(%d) different than the number of targets (%d)"
                    raise Exception(msg % (len(self.inputs.in_sources), len(self.inputs.in_targets)))
                else:
                    for index, paramObj in enumerate(self.inputs.in_paramObjects):
                        arg += '%s %s %s ' % (paramObj, self.inputs.in_sources[index], self.inputs.in_targets[index])
                    return arg
        else:
            return super(SparseMatching3, self)._format_arg(opt, spec, val)


class decimateVTKfileInputSpec(CommandLineInputSpec):
    in_file = File(argstr="-i %s", exists=True, mandatory=True,
                   desc="Input vtk filename")
    out_file = File(argstr="-o %s",
                    desc="output mesh file in VTK format",
                    name_source=['in_file'],
                    name_template='%s_mesh.vtk',
                    keep_extension=False)
    in_reductionRate = traits.Float(0.2,
                                    argstr="-r %s",
                                    mandatory=False,
                                    desc='rate of mesh reduction: if r=0.20: 20pc reduction' +
                                         '(if there was 100 triangles, now there will be 80)',
                                    use_default=True)


class decimateVTKfileOutputSpec(TraitedSpec):
    out_file = File(desc="Output mesh file in VTK format", exists=True)


class decimateVTKfile(DeformetricaCommand):
    """
    Examples
    --------
    """
    _cmd = "decimateVTKfile"
    input_spec = decimateVTKfileInputSpec
    output_spec = decimateVTKfileOutputSpec


class ShootAndFlow3InputSpec(CommandLineInputSpec):
    in_paramDiffeo = File(argstr="%s", exists=True, mandatory=True, position=0,
                          desc='xml file containing the parameters for the diffeo computation ')
    in_direction = traits.Int(1, usedefault=True, position=1,
                              desc='Direction = -1 for using the inverse flow, +1 otherwise')
    in_cp_file = File(exists=True, mandatory=True, position=2,
                      desc='CP.txt file, containing the initial position of the control points.')
    in_mom_file = File(exists=True, mandatory=True, position=3,
                       desc='MOM.txt file, containing the initial momentum of the control points.')

    in_paramObjects = InputMultiPath(exists=True, mandatory=True, position=-1,
                                     desc='xml files containing the parameters of objects ')
    in_sources = InputMultiPath(exists=True, mandatory=True,
                                desc='vtk files containing the mesh of the source of each object. ' +
                                'There is the same number of in_paramObjects than sources. ' +
                                'works only for 1 object at the moment.')


class ShootAndFlow3OutputSpec(TraitedSpec):
    out_files_vtk = traits.List(File(exists=True),
                                            desc="a set of files of the same format as the kth input " +
                                                 "object file. These files give the deformation of " +
                                                 "the object along the optimal deformation")
    out_CP_files_txt = traits.List(File(),
                                        desc='CP txt files of the shooting.')
    out_CP_last_txt = File( desc='last CP txt file of the shooting.')
    out_last_vtk = File( desc='last CP vtk file of the shooting.')


class ShootAndFlow3(DeformetricaCommand):
    """ This command allows the registration between two sets of shapes. A set of shapes is made of the union of
    different objects (Object1, Object2, etc..) embedded in the ambient 2D or 3D space.
    """
    _cmd = "ShootAndFlow3"
    input_spec = ShootAndFlow3InputSpec
    output_spec = ShootAndFlow3OutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        f = open(self.inputs.in_paramDiffeo, 'r')
        line = f.readline()
        while '<number-of-timepoints>' not in line:
            line = f.readline()

        nb_tp = line[line.find('<number-of-timepoints>')+len('<number-of-timepoints>'):
                     line.find('</number-of-timepoints>')]
        print('Number of Time points: ' + str(nb_tp))

        # Works only for 1 object:
        res = self._gen_output_filename(self, self.inputs.in_sources[0],nb_tp)
        outputs['out_files_vtk'] = res[0]
        outputs['out_last_vtk'] = res[1]
        p = os.getcwd()
        outputs['out_CP_files_txt'] = [p + '/' + 'CP_t_' + str(i) + '.txt'
                                       for i in range(int(nb_tp))]
        outputs['out_CP_last_txt'] = p + '/' + 'CP_t_' + str(int(nb_tp)-1) + '.txt'
        return outputs

    @staticmethod
    def _gen_output_filename(s, in_file, nb_tp):
        p, bn, ext = split_filename(in_file)
        list_outfile = [p + '/' + bn + '_flow__t_' + str(i) + '.vtk'
                        for i in range(int(nb_tp))]
        last_outfile = p + '/' + bn + '_flow__t_' + str(int(nb_tp)-1) + '.vtk'
        outfile = [list_outfile, last_outfile]
        return outfile

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for ShootAndFlow3."""
        print " =====debug===== " + opt
        if opt == 'in_paramDiffeo':
            # Write the input as a list of trait.Files
            arg = ' ' + self.inputs.in_paramDiffeo + ' ' + str(self.inputs.in_direction) + ' ' + \
                  self.inputs.in_cp_file + ' ' + self.inputs.in_mom_file + ' '
            if len(self.inputs.in_paramObjects) != len(self.inputs.in_sources):
                # error
                msg = "Number of in_paramObjects (%d) different than the number of sources (%d)"
                raise Exception(msg % (len(self.inputs.in_paramObjects), len(self.inputs.in_sources)))
            else:

                for index, paramObj in enumerate(self.inputs.in_paramObjects):
                    arg += '%s %s ' % (paramObj, self.inputs.in_sources[index])
                return arg
        else:
            return super(ShootAndFlow3, self)._format_arg(opt, spec, val)


class WriteXMLFilesInputSpec(BaseInterfaceInputSpec):
    # import datetime
    # now = datetime.datetime.now()
    today_date = 'today' #now.strftime("%Y_%b_%d") # %Y_%b_%dT%Hh%M
    type_xml_file = traits.Enum('All', 'Def', 'Obj', usedefault=True,
                                desc="'Def' to write the deformation xml file. " +
                                     "'Obj' for the object parameter xml file. " +
                                     "'All' to write both files",
                                mandatory=True)
    path = traits.String(desc=" path where the xml_files folder will be created.")
    dkw = traits.Int(desc=' Diffeo Kernel width ',
                     default_value=10)
    dkt = traits.String("Exact", desc=' Diffeo Kernel type ',
                        usedefault=True)
    dtp = traits.Int(desc=' Diffeo: number of time points',
                     default_value=30)
    dsk = traits.Float(desc=' Diffeo: smoothing kernel width ',
                       default_value=0.5)
    dcps = traits.Int(desc="Diffeos: Initial spacing for Control Points",
                      usedefault=True, default_value=5)
    dcpp = File('x', desc="Diffeos: name of a file containing positions of control points. In case of conflict with " +
                          "initial-cp-spacing, if a file name is given in initial-cp-position and initial-cp-spacing " +
                          "is set, the latter is ignored and control point positions in the file name are used.",
                usedefault=True)
    dfcp = traits.String('Off', desc="Diffeos: Freeze the Control Points",
                         usedefault=True)
    dmi = traits.Int(desc="Diffeos : Maximum of descent iterations", usedefault=True,
                     default_value=100)
    dat = traits.Float(desc="Diffeos : adaptative tolerence for the gradient descent", usedefault=True,
                        default_value=0.00005)
    dls = traits.Int(desc="Diffeos: Maximum line search iterations", usedefault=True,
                     default_value=20)

    ods = traits.ListFloat(desc="Object: weight of the object in the fidelity-to-data term ", usedefault=True,
                           default_value=[0.5])
    okw = traits.ListInt(default_value=[3], desc="Object: Kernel width", usedefault=True)
    ot = traits.List(["NonOrientedSurfaceMesh"], desc="Object type",
                     usedefault=True)
    xml_diffeo = traits.String('parametersDiffeo' + today_date + '.xml',
                               desc='Name of the xml file containing the diffeo parameters', usedefault=True
                               )


class WriteXMLFilesOutputSpec(TraitedSpec):
    out_xmlDiffeo = File(desc="xml file containing the parameters for the diffeomorphism")
    out_xmlObject = OutputMultiPath(desc="xml file(s) containing the parameters of the objects to be deformed")


# XML_TEMP = """
# <?xml...>
# <kkkk>
# <kkk> {dkw}
# <kk>
# {x}
# """
# XML_TEMP.format(dkw=dkw, ....)
class WriteXMLFiles(BaseInterface):
    input_spec = WriteXMLFilesInputSpec
    output_spec = WriteXMLFilesOutputSpec

    def _run_interface(self, runtime):
        import datetime

        if self.inputs.type_xml_file == 'Def' or self.inputs.type_xml_file == 'All':

            if isdefined(self.inputs.path):
                path = self.inputs.path
            else:
                print " =============================== DEBUG ! ++++++++++++++++++++++++++"
                print str(os.getcwd())
                path = str(os.getcwd())

            dkw = self.inputs.dkw
            dkt = self.inputs.dkt
            dtp = self.inputs.dtp
            dsk = self.inputs.dsk
            dcps = self.inputs.dcps
            dcpp = self.inputs.dcpp
            dfcp = self.inputs.dfcp
            dmi = self.inputs.dmi
            dat = self.inputs.dat
            dls = self.inputs.dls
            ods = self.inputs.ods
            okw = self.inputs.okw
            ot = self.inputs.ot
            xml_diffeo = self.inputs.xml_diffeo

            now = datetime.datetime.now()
            today_date = now.strftime("%Y_%b_%dT%Hh%M")

            if not os.path.exists(os.path.join(str(path), 'xml_files/')):
                os.makedirs(path + '/xml_files/')

            path = str(path) + '/xml_files/'
            print "======DEBUG: " + path + xml_diffeo

            self._out_xmlObject = []
            self._out_xmlDiffeo = os.path.abspath(os.path.join(path, xml_diffeo))
            file_param_diffeos = open(self._out_xmlDiffeo, "w")
            file_param_diffeos.write('<?xml version="1.0"?> \n')
            file_param_diffeos.write('<sparse-diffeo-parameters> \n')
            file_param_diffeos.write('<!-- Size of the kernel (default : 10.0) --> \n')
            file_param_diffeos.write('<kernel-width>' + str(dkw) + '</kernel-width> \n')
            file_param_diffeos.write('<!-- Choice of the evaluation method of the kernel : exact, fgt, p3m ' +
                                     '(default : Exact) --> \n')
            file_param_diffeos.write('<kernel-type>' + dkt + '</kernel-type> \n')
            file_param_diffeos.write('<!-- Choice of the number of time points between 0 and 1 (default : 30) --> \n')
            file_param_diffeos.write('<number-of-timepoints>' + str(dtp) + '</number-of-timepoints> \n')
            file_param_diffeos.write('<!-- ratio between the smoothing kernel width used in the gradient of ' +
                                     'template gradient and kernel width (default: 1.0)  --> \n')
            file_param_diffeos.write('<smoothing-kernel-width-ratio>' + str(dsk) + '</smoothing-kernel-width-ratio> \n')
            if dcpp != 'x':
                file_param_diffeos.write('<!-- name of a file containing positions of control points. ' +
                                         'In case of conflict with initial-cp-spacing. If a file name is given in ' +
                                         'initial-cp-position and initial-cp-spacing is set, the latter is ignored ' +
                                         'and control point positions in the file name are used. (default : void) --> \n')
                file_param_diffeos.write('<initial-cp-position>' + str(dcpp) + '</initial-cp-position> \n')

            file_param_diffeos.write('<!-- Step of the regular lattice of control points (default : kernel-width) --> \n')
            file_param_diffeos.write('<initial-cp-spacing>' + str(dcps) + '</initial-cp-spacing> \n')
            file_param_diffeos.write('<!-- Enables to freeze the control points or not (default : Off) --> \n')
            file_param_diffeos.write('<freeze-cp>' + dfcp + '</freeze-cp> \n')
            file_param_diffeos.write('<!-- Maximum of descent iterations (default : 100) --> \n')
            file_param_diffeos.write('<max-iterations>' + str(dmi) + '</max-iterations> \n')
            file_param_diffeos.write('<!-- Adaptive tolerance (default : 1e-04) --> \n')
            file_param_diffeos.write('<adaptive-tolerance>' + str(dat) + '</adaptive-tolerance> \n')
            file_param_diffeos.write('<!-- Maximum number for the search of the optimal step on an iteration ' +
                                     '(default : 10) --> \n')
            file_param_diffeos.write('<max-line-search-iterations>' + str(dls) + '</max-line-search-iterations> \n')
            file_param_diffeos.write('<atlas-type>Bayesian</atlas-type>')
            file_param_diffeos.write('</sparse-diffeo-parameters> \n')
            file_param_diffeos.close()

        if self.inputs.type_xml_file == 'Obj' or self.inputs.type_xml_file == 'All':
            if (len(okw) is not len(ods)) or (len(ot) is not len(ods)) or (len(okw) is not len(ot)):
                raise Exception('Writing the xml files. Need to have one value per object for ' +
                                'the values ods okw and ot. So they need to have the same size (=nb of objects).' +
                                'The number of input images and input GIF parcelation should be identical. Here:'
                                ' ods length= ' + str(len(ods)) + ' okw length = ' + str(len(okw)) +
                                ' ot length= ' + str(len(ot)))

            for i in range(len(okw)):
                if i == 0:
                    self._out_xmlObject = [os.path.abspath(os.path.join(path, 'paramObjet_' +str(i+1) + '_' + today_date + '.xml'))]
                else:
                    self._out_xmlObject.append(os.path.abspath(os.path.join(path, 'paramObjet_' +str(i+1) + '_' + today_date + '.xml')))

                file_param_objects = open(self._out_xmlObject[i], "w")
                file_param_objects.write('<?xml version="1.0"?> \n')
                file_param_objects.write('<deformable-object-parameters> \n')
                file_param_objects.write('<!-- Type of the deformable object ' +
                                         '(See DeformableObject::DeformableObjectType for details) --> \n')
                file_param_objects.write('<deformable-object-type>' + ot[i] + '</deformable-object-type>  \n')
                file_param_objects.write('<!-- weight of the object in the fidelity-to-data term --> \n')
                file_param_objects.write('<data-sigma>' + str(ods[i]) + '</data-sigma>  \n')
                file_param_objects.write('<!-- Size of the kernel (default : 0.0) --> \n')
                file_param_objects.write('<kernel-width>' + str(okw[i]) + '</kernel-width> \n')
                file_param_objects.write('</deformable-object-parameters> \n')
                file_param_objects.close()

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_xmlDiffeo'] = self._out_xmlDiffeo
        outputs['out_xmlObject'] = self._out_xmlObject
        return outputs


class sortingTimePointsInputSpec(BaseInterfaceInputSpec):
    in_xmlDiffeo = traits.File(exists=True, mandatory=True,
                               desc='xml file used for the computation of the spatio temporal trajectory')
    in_AgeToOnsetNorm_file = traits.File(exists=True, mandatory=True)
    in_timePoints_vtkfile = traits.List(File(exists=True), mandatory=True)
    in_CP_txtfiles = traits.List(File(exists=True), desc='CP files from the shooting')


class sortingTimePointsOutputSpec(TraitedSpec):
    out_t_from = traits.ListInt(desc='list of time points sorted the same way as the 2 other output files.')
    files_trajectory_source = traits.List(File(exists=True,
                                               desc='list of the source.vtk file sorted by the time points'))
    files_xmlDiffeo = traits.List(File,
                                  desc='list of param Diffeo file, to be able to use the CP.txt files generated by ' +
                                       'the shooting as initial CP position for the subsequent ' +
                                       'SparseMatching3 function.')
    subject_id = traits.List(traits.String())

# specific to deformetrica, sort the time points and the corresponding shapes, and write the
# xml files by adding the position of the CP, required afterwards for the SparseMatching3 function.
class sortingTimePoints(BaseInterface):
    input_spec = sortingTimePointsInputSpec
    output_spec = sortingTimePointsOutputSpec

    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    def _run_interface(self, runtime):

        f = open(self.inputs.in_xmlDiffeo, 'r')
        line = f.readline()
        while '<number-of-timepoints>' not in line:
            line = f.readline()

        nb_tp = line[line.find('<number-of-timepoints>') + len('<number-of-timepoints>') :
                     line.find('</number-of-timepoints>')]

        # make sure that the nb_tp is the length of the in_timePoints_vtkfile:
        if nb_tp != str(len(self.inputs.in_timePoints_vtkfile)):
            msg = "Number of tp(%s) different than the number of vtk files from the geodesic shooting (%d)"
            raise Exception(msg % (nb_tp), len(self.inputs.in_timePoints_vtkfile))

        self.files_xmlDiffeo = []
        subjects_id = []
        subject_filename_vtk = []
        ages_norm = []
        self.in_t_from = []
        self.subject_id =[]

        print "===DEBUG=== nb_tp = " + nb_tp
        for line in open(self.inputs.in_AgeToOnsetNorm_file):
            subjects_id.append(line.split()[0])
            subject_filename_vtk.append(line.split()[1])
            ages_norm.append(line.split()[2])


        time_points = range(int(nb_tp))
        tp_normalised = [float(t) / float(nb_tp) for t in time_points]
        # print "===DEBUG=== time_points = " + str(time_points)
        self.files_trajectory_source = []
        # loop on the subjects ID
        for i, suj in enumerate(subjects_id):
            print"== DEBUG ==: i=" + str(i) + " suj = " + suj
            # find the closest time points of the subject
            aa = [abs(float(v) - float(ages_norm[i])) for v in tp_normalised]
            closest_tp_ind = aa.index(min(aa))
            self.in_t_from.append(time_points[closest_tp_ind])
            self.files_trajectory_source.append(self.inputs.in_timePoints_vtkfile[closest_tp_ind])
            self.subject_id.append(suj)
            # creating and writing the xml diffeo files.
            # the index must be the same as for the source file.
            p, name, ext = split_filename(os.path.abspath(self.inputs.in_xmlDiffeo))
            new_xmldiffeo_file = os.path.join(p, name + '_forMatching_' + suj + ext)
            self.files_xmlDiffeo.append(new_xmldiffeo_file)
            in_file = open(self.inputs.in_xmlDiffeo, 'r')
            out_file = open(new_xmldiffeo_file, 'w+')
            check_error = True
            for line in in_file:
                print line
                if ('<initial-cp-position>' in line) or ('<initial-cp-spacing>' in line):
                    out_file.write('<initial-cp-position>' + self.inputs.in_CP_txtfiles[closest_tp_ind] +
                                   '</initial-cp-position> \n')
                    print 'newline: <initial-cp-position>' + self.inputs.in_CP_txtfiles[closest_tp_ind] + '</initial-cp-position> \n'
                    check_error = False
                elif '</sparse-diffeo-parameters> ' in line:
                    out_file.write('<initial-cp-position>' + self.inputs.in_CP_txtfiles[closest_tp_ind] +
                                   '</initial-cp-position> \n')
                    out_file.write('</sparse-diffeo-parameters> \n')
                    check_error = False
                else:
                    out_file.write(line)
            in_file.close()
            out_file.close()
            if check_error:
                raise ValueError('sortingTimePoints interface: the given paramDiffeo.xml file is wrong.' +
                                         'It should contain at least "</sparse-diffeo-parameters>" ')

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['files_trajectory_source'] = self.files_trajectory_source
        outputs['out_t_from'] = self.in_t_from
        outputs['files_xmlDiffeo'] = self.files_xmlDiffeo
        outputs['subject_id'] = self.subject_id
        return outputs
