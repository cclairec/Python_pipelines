# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import pickle
import scipy.io
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec, isdefined, TraitedSpec, File, traits)
from nipype.interfaces.matlab import MatlabCommand, MatlabCommand, MatlabInputSpec
from nipype.utils.filemanip import split_filename
from string import Template

from .base import ShapeInputSpec, ShapeMatlabInputSpec


class ComputeBarycentreBaseLineInputSpec(ShapeInputSpec):
    in_initId = traits.Int(-1, usedefault=True,
                           desc='-1 for a random order. x to start with the subject number x.')
    in_subjects = File(exists=True,
                       desc='.mat file containing cells of structures. One cell per subject, with:'
                            'surf_aligned{k}.Vertices = 3xNbVertices'
                            'surf_aligned{k}.Faces = 3xNbFaces ',
                       mandatory=True)
    in_pct = traits.Int(-1, usedefault=True,
                        desc='use x% of the population to compute the centroid. if -1, use 100pc of the population')
    in_docombi = traits.Int(default_value=0, usedefault=True,
                            desc='If 1, use the algo IC2 (use the reverse flow). If 0, use the algo IC1.')

    in_param_gammaR = traits.Float(1.e-4, usedefault=True)
    in_param_sigmaV = traits.Float(13, usedefault=True)
    in_param_sigmaW = traits.List(traits.Float, [11, 8, 4, 2], usedefault=True)
    in_param_maxiters = traits.ListFloat(default_value=[100, 200, 200, 100], usedefault=True)
    in_param_T = traits.Float(default_value=10, usedefault=True)
    in_param_ntries = traits.Float(default_value=1, usedefault=True)
    in_param_MPeps = traits.Float(default_value=0, usedefault=True)
    out_suffix = traits.String('', usedefault=True)
    script = traits.String(desc='matlab script, defined later in the class. ' +
                                'The input script is asked by the MatlabCommand class.')


class ComputeBarycentreBaseLineOutputSpec(TraitedSpec):
    out_vertices_file = File(desc='.txt file containing the vertices of the centroid.')
    out_file = File(desc='.mat file where the results are saved.')


class ComputeBarycentreBaseLine(BaseInterface):
    input_spec = ComputeBarycentreBaseLineInputSpec
    output_spec = ComputeBarycentreBaseLineOutputSpec

    def _run_interface(self, runtime):
        print "======= DEBUG ComputeBarycentreBaseLine ============="
        output = self._list_outputs()
        print " do combi: " + str(self.inputs.in_docombi)
        print output['out_vertices_file']
        print output['out_file']

        d = dict(path_matlab=self.inputs.path_matlab,
                 in_initId=self.inputs.in_initId,
                 in_subjects=self.inputs.in_subjects,
                 in_pct=self.inputs.in_pct,
                 in_docombi=self.inputs.in_docombi,
                 in_param_gammaR=self.inputs.in_param_gammaR,
                 in_param_sigmaV=self.inputs.in_param_sigmaV,
                 in_param_sigmaW=self.inputs.in_param_sigmaW,
                 in_param_maxiters=self.inputs.in_param_maxiters,
                 in_param_T=self.inputs.in_param_T,
                 in_param_ntires=self.inputs.in_param_ntries,
                 in_param_MPeps=self.inputs.in_param_MPeps,
                 out_vertices=output['out_vertices_file'],
                 out_filename=output['out_file']
                 )

        self.inputs.script = Template("""

            pathData = pwd;
            addpath('$path_matlab');
            matlab_startup;
            param.gammaR=$in_param_gammaR;
            param.sigmaV=$in_param_sigmaV;
            param.sigmaW=$in_param_sigmaW;
            param.maxiters=$in_param_maxiters;
            param.T=$in_param_T;
            param.ntries=$in_param_ntires;
            filename='$in_subjects'
            param.MPeps=$in_param_MPeps;
            sname=load('$in_subjects');
            data_temp = struct2cell(sname);
            surf_aligned=data_temp{1};
            CP_initFile = '$out_vertices'
            fileOut = '$out_filename';
            init = $in_initId;
            pct = $in_pct;
            doCombi = $in_docombi;
            for i=1:length(surf_aligned)
                age(i)=surf_aligned{1,i}.Age;
            end
            minAge=min(age);
            maxAge=max(age);
            age=(age-minAge)./(maxAge-minAge);
            %ageToOnsetNorm_file = [pathData '/AgeToOnsetNorm.txt'];

            %fid = fopen(out_ageToOnsetNorm_file, 'w');
            %for i=1:length(surf_aligned)
            %    %surf_aligned{1,i}.Age_Norm=age(i);
            %    fprintf(fid,[surf_aligned{1,i}.SujID ' ' surf_aligned{1,i}.Filename ' ' num2str(age(i)) char(10)])
            %end
            %fclose(fid);

            [Bary, B ] = BarycentreSurface(init, surf_aligned, fileOut, pct, doCombi, param);

            %Bary{1}.Vertices = zeros(3,100);
            %Bary{1}.Faces = round(rand(2,100)*100+1)
            save(fileOut, 'Bary');

            fid = fopen(CP_initFile, 'w');
            for i=1: length(Bary{1}.Vertices)
                fprintf(fid,[num2str(Bary{1}.Vertices(1,i)) ' '...
                    num2str(Bary{1}.Vertices(2,i)) ' '...
                    num2str(Bary{1}.Vertices(3,i)) char(10)]);
            end
            fclose(fid);
            exit;

            warning('Something went wrong in the matlab script of the interface ComputeBarycentreBaseLine');

        exit;
        """).substitute(d)

        # mfile = True  will create an .m file with your script and executed.
        # Alternatively
        # mfile can be set to False which will cause the matlab code to be
        # passed
        # as a commandline argument to the matlab executable
        # (without creating any files).
        # This, however, is less reliable and harder to debug
        # (code will be reduced to
        # a single line and stripped of any comments).
        mlab = MatlabCommand(script=self.inputs.script, mfile=True)
        results = mlab.run()
        return results.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        _, name, ext = split_filename(self.inputs.in_subjects)
        outputs['out_file'] = os.path.abspath(name + self.inputs.out_suffix + ext)
        outputs['out_vertices_file'] = os.path.abspath(name + self.inputs.out_suffix + '_vertices.txt')
        return outputs


class VTKPolyDataReaderInputSpec(ShapeInputSpec):
    # one of the two is mandatory
    in_filenames = traits.List(desc='VTK file names', xor=['in_filename'])
    in_filename = File(desc='VTK file name', xor=['in_filenames'])
    in_struct = File('Empty',
                     desc='file.mat containing the matlab structure surf_aligned, which is missing the fields'
                          ' .Vertices and .Faces',
                     usedefault=True)
    script = traits.String(desc='matlab script, defined later in the class. ' +
                                'The input script is askinf by the MatlabCommand class.')
    symmetric = traits.Bool(False, desc='if data are being symmetrised, True will allow the creation of L/R list ' +
                                        'of structure, one for each subject for the next centroid computation step. ',
                            usedefault=True)


class VTKPolyDataReaderOutputSpec(TraitedSpec):
    out_verticesFile = File(desc='.txt file containing the vertices of vtk file.')
    out_triangleFile = File(desc='.txt file containing the triangles of the vtk file.')
    out_verticesFiles = traits.List(desc='.txt file containing the vertices of vtk file.')
    out_triangleFiles = traits.List(desc='.txt file containing the triangles of the vtk file.')
    out_structFile = File(desc='.mat file updated version the input in_struct file.')
    out_structListFile = traits.List(File(desc='.mat file updated version the input in_struct file.'))


class VTKPolyDataReader(BaseInterface):
    input_spec = VTKPolyDataReaderInputSpec
    output_spec = VTKPolyDataReaderOutputSpec

    # def _my_script(self):
    #     """This is where you implement your script"""
    #     output = self._list_outputs()
    #     files=[]
    #     if not isdefined(self.inputs.in_filenames):
    #         files.append(self.inputs.in_filename)
    #         tri_file=output['out_verticesFile']
    #         points_file=output['out_triangleFile']
    #     else:
    #         files = self.inputs.in_filenames
    #
    #     if isdefined(self.inputs.in_filenames):
    #         tri_file=output['out_verticesFiles']
    #         points_file=output['out_triangleFiles']
    #
    #     print "======DEBUG VTKPolyDataReader ======== "
    #     print "===" + self.inputs.path_matlab
    #     d = dict(path_matlab=self.inputs.path_matlab,
    #              in_filenames=files,
    #              in_struct=self.inputs.in_struct,
    #              tri_files=tri_file,
    #              points_files=points_file,
    #              symmetric=self.inputs.symmetric)
    #     script = Template("""display('Hello ! Matlab started!');
    #     path_ = pwd;
    #     addpath('$path_matlab');
    #     matlab_startup;
    #     filenames='$in_filenames'
    #     struct='$in_struct'
    #     Points_files='$points_files'
    #     Tri_files='$tri_files'
    #     ind_beg_filepath = findstr(filenames,path_(1:6));
    #     ind_beg_tripath = findstr(tri_files,path_(1:6));
    #     ind_beg_pointspath = findstr(points_files,path_(1:6));
    #     for ind = 1: length(ind_beg_filepath)
    #         clear filename;
    #         if ind == ind_beg_filepath(end)
    #         filename = filenames(ind_beg_filepath(ind):end);
    #         else
    #         filename = filenames(ind_beg_filepath(ind):ind_beg_filepath(ind+1));
    #         end
    #         [Points, Tri] = VTKPolyDataReader(filename);
    #         if strcmp(struct,'Empty')
    #             display('VTKPolyDataReader: just save the .txt files, no structure.mat file given')
    #         elseif ind>0 & strcmp(struct(end-3:end),'.mat')
    #               sname=load(struct);
    #               data_temp = struct2cell(sname);
    #               surf_aligned=data_temp{1};
    #               if ~strcmp(surf_aligned{ind}.Filename,filename)
    #                 error(['filename does not match: surf_aligned{ind}.Filename = ' surf_aligned{ind}.Filename ', filename = ' filename '.']);
    #               else
    #                 surf_aligned{ind}.Vertices = Points';
    #                 surf_aligned{ind}.Faces = Tri';
    #                 save(struct,'surf_aligned');
    #               end
    #         else
    #               error(['VTKPolyDataReader: Check the inputs: struct = ' struct ', filename = ' filename ', ind = ' ind '.']);
    #               exit;
    #         end
    #         % converting the Point and Tri variables in .txt files
    #         clear Points_file;
    #         if ind == ind_beg_pointspath(end)
    #             Points_file = Points_files(ind_beg_pointspath(ind):end);
    #         else
    #             Points_file = Points_files(ind_beg_pointspath(ind):ind_beg_pointspath(ind+1));
    #         end
    #         fid = fopen(Points_file, 'w');
    #         for i=1: length(Points)
    #             fprintf(fid,[num2str(Points(i,1)) ' '...
    #                 num2str(Points(i,2)) ' '...
    #                 num2str(Points(i,3)) char(10)]);
    #         end
    #         fclose(fid);
    #         clear fid;
    #
    #         clear Tri_file;
    #         if ind == ind_beg_tripath(end)
    #             Tri_file = Tri_files(ind_beg_tripath(ind):end);
    #         else
    #             Tri_file = Tri_files(ind_beg_tripath(ind):ind_beg_tripath(ind+1));
    #         end
    #         fid = fopen(Tri_file, 'w');
    #         for i=1: length(Tri)
    #             fprintf(fid,[num2str(Tri(i,1)) ' '...
    #                 num2str(Tri(i,2)) ' '...
    #                 num2str(Tri(i,3)) char(10)]);
    #         end
    #         fclose(fid);
    #     end
    #     exit;
    #     """).substitute(d)
    #     return script
    #
    #
    # def run(self, **inputs):
    #     ## inject your script
    #
    #     self.inputs.script=self._my_script()
    #     results=super(MatlabCommand, self).run( **inputs)
    #     print results
    #     stdout = results.runtime.stdout
    #     # attach stdout to outputs to access matlab results
    #     results.outputs.matlab_output = stdout
    #     return results

    def _run_interface(self, runtime):

        output = self._list_outputs()
        files=[]
        if not isdefined(self.inputs.in_filenames):
            files.append(self.inputs.in_filename)
            tri_files=output['out_verticesFile']
            points_files=output['out_triangleFile']
        else:
            files = self.inputs.in_filenames
            tri_files=output['out_verticesFiles']
            points_files=output['out_triangleFiles']

        print "======DEBUG VTKPolyDataReader ======== "
        print "===" + self.inputs.path_matlab
        d = dict(path_matlab=self.inputs.path_matlab,
                 in_filenames=files,
                 in_struct=self.inputs.in_struct,
                 tri_files=tri_files,
                 points_files=points_files,
                 symmetric=self.inputs.symmetric)
        script = Template("""display('Hello ! Matlab started!');
        path_ = pwd;
        addpath('$path_matlab');
        matlab_startup;
        filenames=$in_filenames
        struct='$in_struct'
        Points_files=$points_files
        Tri_files=$tri_files
        ind_beg_filepath = findstr(filenames,path_(1:6));
        ind_beg_tripath = findstr(Tri_files,path_(1:6));
        ind_beg_pointspath = findstr(Points_files,path_(1:6));
        for ind = 1: length(ind_beg_filepath)
            clear filename;
            if ind == length(ind_beg_filepath)
            filename = filenames(ind_beg_filepath(ind):end);
            else
            filename = filenames(ind_beg_filepath(ind):ind_beg_filepath(ind+1)-1);
            end
            [Points, Tri] = VTKPolyDataReader(filename);
            if strcmp(struct,'Empty')
                display('VTKPolyDataReader: just save the .txt files, no structure.mat file given')
            elseif ind>0 & strcmp(struct(end-3:end),'.mat')
                  sname=load(struct);
                  data_temp = struct2cell(sname);
                  surf_aligned=data_temp{1};
                  if ~strcmp(surf_aligned{ind}.Filename,filename)
                    error(['filename does not match: surf_aligned{ind}.Filename = ' surf_aligned{ind}.Filename ', filename = ' filename '.']);
                  else
                    surf_aligned{ind}.Vertices = Points';
                    surf_aligned{ind}.Faces = Tri';
                    save(struct,'surf_aligned');
                  end
            else
                  error(['VTKPolyDataReader: Check the inputs: struct = ' struct ', filename = ' filename ', ind = ' ind '.']);
                  exit;
            end
            % converting the Point and Tri variables in .txt files
            clear Points_file;
            if ind == length(ind_beg_pointspath)
                Points_file = Points_files(ind_beg_pointspath(ind):end);
            else
                Points_file = Points_files(ind_beg_pointspath(ind):ind_beg_pointspath(ind+1)-1);
            end
            fid = fopen(Points_file, 'w');
            for i=1: length(Points)
                fprintf(fid,[num2str(Points(i,1)) ' '...
                    num2str(Points(i,2)) ' '...
                    num2str(Points(i,3)) char(10)]);
            end
            fclose(fid);
            clear fid;

            clear Tri_file;
            if ind == length(ind_beg_tripath)
                Tri_file = Tri_files(ind_beg_tripath(ind):end);
            else
                Tri_file = Tri_files(ind_beg_tripath(ind):ind_beg_tripath(ind+1)-1);
            end
            fid = fopen(Tri_file, 'w');
            for i=1: length(Tri)
                fprintf(fid,[num2str(Tri(i,1)) ' '...
                    num2str(Tri(i,2)) ' '...
                    num2str(Tri(i,3)) char(10)]);
            end
            fclose(fid);
        end
        exit;
        """).substitute(d)
            # mfile = True  will create an .m file with your script and executed.
            # Alternatively
            # mfile can be set to False which will cause the matlab code to be
            # passed
            # as a commandline argument to the matlab executable
            # (without creating any files).
            # This, however, is less reliable and harder to debug
            # (code will be reduced to
            # a single line and stripped of any comments).
        self.script = script
        #result = super(VTKPolyDataReader, self)._run_interface(runtime)
        mlab = MatlabCommand(script=script, mfile=True)
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_verticesFiles'] = []
        outputs['out_triangleFiles'] = []
        files=[]
        if not isdefined(self.inputs.in_filenames):
            files.append(self.inputs.in_filename)
            outputs['out_verticesFile']=self._gen_output_filename(files, '_points.txt')
            outputs['out_triangleFile']=self._gen_output_filename(files, '_tri.txt')
        elif len(self.inputs.in_filenames) == 1:
            files.append(self.inputs.in_filenames)
            outputs['out_verticesFile']=self._gen_output_filename(files, '_points.txt')
            outputs['out_triangleFile']=self._gen_output_filename(files, '_tri.txt')

        if isdefined(self.inputs.in_filenames):
            files = self.inputs.in_filenames
            for i in range(len(files)):
                print i
                outputs['out_verticesFiles'].append(self._gen_output_filename(files, '_points.txt', i))
                outputs['out_triangleFiles'].append(self._gen_output_filename(files, '_tri.txt', i))
                if self.inputs.symmetric:
                    outputs['out_structListFile'].append(self._gen_output_filename(self.inputs.in_struct, 'LR_suj', i))
            if isdefined(self.inputs.in_struct) & (not self.inputs.symmetric):
                outputs['out_structFile'] = os.path.abspath(self.inputs.in_struct)

        return outputs

    @staticmethod
    def _gen_output_filename(in_files, ext, ind=0):
        _, bn, _ = split_filename(in_files[ind])
        outfile = os.path.abspath(bn + ext)
        return outfile



# class VTKPolyDataReader(BaseInterface):
#     input_spec = VTKPolyDataReaderInputSpec
#     output_spec = VTKPolyDataReaderOutputSpec
#
#     def _run_interface(self, runtime):
#
#         output = self._list_outputs()
#         files=[]
#         if not isdefined(self.inputs.in_filenames):
#             files.append(self.inputs.in_filename)
#             tri_file=output['out_verticesFile']
#             points_file=output['out_triangleFile']
#         else:
#             files = self.inputs.in_filenames
#
#         for i in range(len(files)):
#             if isdefined(self.inputs.in_filenames):
#                 tri_file=output['out_verticesFiles'][i]
#                 points_file=output['out_triangleFiles'][i]
#
#             file_name = files[i]
#             print "======DEBUG VTKPolyDataReader ======== " + str(i)
#             d = dict(path_matlab=self.inputs.path_matlab,
#                      in_filenames=file_name,
#                      in_struct=self.inputs.in_struct,
#                      tri_file=tri_file,
#                      points_file=points_file,
#                      in_ind=i+1,
#                      symmetric=self.inputs.symmetric)
#
#             #this is your MATLAB code template
#             self.inputs.script = Template("""
#             path_ = pwd;
#             addpath('$path_matlab');
#             matlab_startup;
#             filename='$in_filenames'
#             struct='$in_struct'
#             ind=str2num('$in_ind')
#             Points_file='$points_file'
#             Tri_file='$tri_file'
#             [Points, Tri] = VTKPolyDataReader(filename);
#             if strcmp(struct,'Empty')
#                 display('VTKPolyDataReader: just save the .txt files, no structure.mat file given')
#             elseif ind>0 & strcmp(struct(end-3:end),'.mat')
#                   sname=load(struct);
#                   data_temp = struct2cell(sname);
#                   surf_aligned=data_temp{1};
#                   if ~strcmp(surf_aligned{ind}.Filename,filename)
#                     error(['filename does not match: surf_aligned{ind}.Filename = ' surf_aligned{ind}.Filename ', filename = ' filename '.']);
#                   else
#                     surf_aligned{ind}.Vertices = Points';
#                     surf_aligned{ind}.Faces = Tri';
#                     save(struct,'surf_aligned');
#                   end
#             else
#                   error(['VTKPolyDataReader: Check the inputs: struct = ' struct ', filename = ' filename ', ind = ' ind '.']);
#                   exit;
#             end
#             % converting the Point and Tri variables in .txt files
#
#             fid = fopen(Points_file, 'w');
#             for i=1: length(Points)
#                 fprintf(fid,[num2str(Points(i,1)) ' '...
#                     num2str(Points(i,2)) ' '...
#                     num2str(Points(i,3)) char(10)]);
#             end
#             fclose(fid);
#             clear fid;
#
#             fid = fopen(Tri_file, 'w');
#             for i=1: length(Tri)
#                 fprintf(fid,[num2str(Tri(i,1)) ' '...
#                     num2str(Tri(i,2)) ' '...
#                     num2str(Tri(i,3)) char(10)]);
#             end
#             fclose(fid);
#
#             exit;
#             """).substitute(d)
#
#             # mfile = True  will create an .m file with your script and executed.
#             # Alternatively
#             # mfile can be set to False which will cause the matlab code to be
#             # passed
#             # as a commandline argument to the matlab executable
#             # (without creating any files).
#             # This, however, is less reliable and harder to debug
#             # (code will be reduced to
#             # a single line and stripped of any comments).
#             mlab = MatlabCommand(script=self.inputs.script, mfile=True)
#             result = mlab.run()
#         return result.runtime
#
#     def _list_outputs(self):
#         outputs = self._outputs().get()
#         outputs['out_verticesFiles'] = []
#         outputs['out_triangleFiles'] = []
#         files=[]
#         if not isdefined(self.inputs.in_filenames):
#             files.append(self.inputs.in_filename)
#             outputs['out_verticesFile']=self._gen_output_filename(files, '_points.txt')
#             outputs['out_triangleFile']=self._gen_output_filename(files, '_tri.txt')
#         elif len(self.inputs.in_filenames) == 1:
#             files.append(self.inputs.in_filenames)
#             outputs['out_verticesFile']=self._gen_output_filename(files, '_points.txt')
#             outputs['out_triangleFile']=self._gen_output_filename(files, '_tri.txt')
#
#         if isdefined(self.inputs.in_filenames):
#             files = self.inputs.in_filenames
#             for i in range(len(files)):
#                 print i
#                 outputs['out_verticesFiles'].append(self._gen_output_filename(files, '_points.txt', i))
#                 outputs['out_triangleFiles'].append(self._gen_output_filename(files, '_tri.txt', i))
#                 if self.inputs.symmetric:
#                     outputs['out_structListFile'].append(self._gen_output_filename(self.inputs.in_struct, 'LR_suj', i))
#             if isdefined(self.inputs.in_struct) & (not self.inputs.symmetric):
#                 outputs['out_structFile'] = os.path.abspath(self.inputs.in_struct)
#
#         return outputs
#
#     @staticmethod
#     def _gen_output_filename(in_files, ext, ind=0):
#         _, bn, _ = split_filename(in_files[ind])
#         outfile = os.path.abspath(bn + ext)
#         return outfile


class VTKPolyDataWriterInputSpec(ShapeInputSpec):
    in_filename = File(desc='.mat file name containing cells of structures: obj{i}.Faces obj{i}.Vertices',
                       mandatory=True)
    script = traits.String(desc='matlab script, defined later in the class. ' +
                                'The input script is asked by the MatlabCommand class.')
    nb_meshes = traits.Int(1, desc='number of meshes in the matlab structure given in in_filename',
                           usedefault=True)


class VTKPolyDataWriterOutputSpec(TraitedSpec):
    # When using this interface, need to use the right output, depending if we want a list (nb Sub >1) or not
    out_filename_list = traits.List(File(exists=True,
                                         desc='.vtk file(s) containing the mesh of the cell(s) from the .mat file.'))
    out_filename = traits.File(exists=True,
                               desc='.vtk file(s) containing the mesh of the cell(s) from the .mat file.')


class VTKPolyDataWriter(BaseInterface):
    input_spec = VTKPolyDataWriterInputSpec
    output_spec = VTKPolyDataWriterOutputSpec
    _suffix = '_fromMatfile'

    def _run_interface(self, runtime):
        print "====== DEBUG VTKPolyDataWriter 1====== "
        output = self._list_outputs()
        nb_suj = self.inputs.nb_meshes
        print "====== DEBUG VTKPolyDataWriter 2====== "
        if nb_suj > 1:
            out_filename =''
            out_filename_list = []
            for i in range(len(output['out_filename_list'])):
                out_filename_list.append(output['out_filename_list'][i])
                print output['out_filename_list'][i]
                out_filename_list.append('char(10)')
        else:
            out_filename = output['out_filename']
            out_filename_list = [out_filename]
        print "====== DEBUG VTKPolyDataWriter 3====== "
        d = dict(path_matlab=self.inputs.path_matlab,
                 in_filename=os.path.abspath(self.inputs.in_filename),
                 out_filename=out_filename,
                 out_filename_list=out_filename_list)
        print "====== DEBUG VTKPolyDataWriter 4====== "
        # this is your MATLAB code template
        self.inputs.script = Template("""
        path_ = pwd;
        addpath('$path_matlab');
        matlab_startup;
        in_filename='$in_filename';
        out_filename='$out_filename';
        out_filename_list=$out_filename_list;
        sname=load(in_filename);
        data_temp = struct2cell(sname);
        data=data_temp{1};

        if length(data) == 1
            points = data{1}.Vertices';
            faces = data{1}.Faces';
            VTKPolyDataWriter(points, faces, [], [], [],out_filename);
        else
            ind_char10=strfind(out_filename,'char(10)')
            ind_beginning_list = 1;
            for i=1:length(data)
                points = data{i}.Vertices';
                faces = data{i}.Faces';
                out_filename_i = [out_filename_list(ind_beginning_list:ind_char10(ind_beginning_list)-5) '.vtk'];
                VTKPolyDataWriter(points, faces, [], [], [],out_filename_i);
                ind_beginning_list = ind_beginning_list+1;
            end
        end
        exit;
        """).substitute(d)
        print "====== DEBUG VTKPolyDataWriter 5====== "

        # mfile = True  will create an .m file with your script and executed.
        # Alternatively
        # mfile can be set to False which will cause the matlab code to be
        # passed
        # as a commandline argument to the matlab executable
        # (without creating any files).
        # This, however, is less reliable and harder to debug
        # (code will be reduced to
        # a single line and stripped of any comments).
        mlab = MatlabCommand(script=self.inputs.script, mfile=True)
        result = mlab.run()
        print "====== DEBUG VTKPolyDataWriter 6====== "
        return result.runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        _, name, ext = split_filename(self.inputs.in_filename)
        nb_suj = self.inputs.nb_meshes
        if nb_suj == 1:
            outputs['out_filename'] = os.path.abspath(name + self._suffix + '_0.vtk')

        list_output = [os.path.abspath(name + self._suffix + '_' + str(i) + '.vtk')
                       for i in range(nb_suj)]
        outputs['out_filename_list'] = list_output
        return outputs


class CreateStructureOfDataInputSpec(BaseInterfaceInputSpec):
    input_meshes = traits.List(File(exists=True))
    subject_ids = traits.List(mandatory=True)
    ages = traits.ListFloat(mandatory=True)
    in_structure_filename = traits.String('structureOfData', usedefault=True,
                                          desc='No need to specify the extension of the file')
    in_label = traits.ListInt(desc='labels of the structures. One label per scan. or one for all the scans ')
    nbsuj_longitudinal = traits.Int(-1, usedefault=True,
                                    desc='if longitudinal data give the number of subjects, ' +
                                         'so a structure file is written for the baseline shapes.')
    out_ageToOnsetNorm_filename = File('AgeToOnsetNorm.txt',
                                       desc='txt file where ages (or time) of each subject are stored',
                                       usedefault=True)
    # symmetric = traits.Bool(False, usedefault=True, desc='When data are symmetrised: meshes are ' +
    #                                                      'from original and flipped image for L and R ' +
    #                                                      'extraction for a common shape computing')


class CreateStructureOfDataOutputSpec(TraitedSpec):
    out_file = traits.File(desc='Output pickle file')
    out_file_mat = traits.File(desc='Output mat file')
    # out_list_file_mat = traits.List(File(desc='Output mat file'))
    out_b0_file_mat = traits.File(desc='Output mat file for baselines bo. ' +
                                       'Only created when the option bool_longitudinal is True.')
    out_b0_file = traits.File(desc='Output pickle file for baselines bo. ' +
                                   'Only created when the option bool_longitudinal is True.')
    out_ageToOnsetNorm_file = File(desc='txt file containing the age to onset normalised for all the subjects.' +
                                        'per line: subject ID, subject vtk filename age.')
    out_ageToOnsetNorm_b0_file = File(desc='txt file containing the age to onset normalised for all the subjects.' +
                                        'per line: subject ID, subject vtk filename age.')


class CreateStructureOfData(BaseInterface):
    input_spec = CreateStructureOfDataInputSpec
    output_spec = CreateStructureOfDataOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        _, name, ext = split_filename(self.inputs.in_structure_filename)
        outputs['out_file'] = os.path.abspath(name + '_' + str(self.inputs.in_label[0]) + '.pkl')
        outputs['out_file_mat'] = os.path.abspath(name + '_' + str(self.inputs.in_label[0]) + '.mat')
        outputs['out_ageToOnsetNorm_file'] = os.path.abspath(self.inputs.out_ageToOnsetNorm_filename)
        # if self.inputs.symmetric:
        #     outputs['out_list_file_mat'] = []
        #     for i in range(self.inputs.in_scan_number):
        #         outputs['out_list_file_mat'].append(self._gen_output_filename(name + '_' + str(self.inputs.in_label[0]), 'LR_suj' + str(i)))

        if self.inputs.nbsuj_longitudinal is not -1:
            outputs['out_b0_file'] = os.path.abspath(name + '_B0_' + str(self.inputs.in_label[0]) + '.pkl')
            outputs['out_b0_file_mat'] = os.path.abspath(name + '_B0_' + str(self.inputs.in_label[0]) + '.mat')
            outputs['out_ageToOnsetNorm_b0_file'] = os.path.abspath(self.output_age_file_b0)

        return outputs

    def _run_interface(self, runtime):

        if len(self.inputs.in_label) > 1:
            if len(self.inputs.in_label)!= len(self.inputs.input_meshes):
                raise ArithmeticError("In creatStructureOfData, we expect the label list size to be 1 or the size of the input scan number")
            else:
                labels = self.inputs.in_label

        elif len(self.inputs.in_label) == 1:
            labels = [self.inputs.in_label[0] for k in range(len(self.inputs.input_meshes))]

        # New GIF labels:
        GIF_left_labels = [31, 33, 38, 40, 42, 46, 49, 51, 53, 57, 59, 61, 63, 76, 106, 138, 148, 180, 144, 164, 166,
                           192, 206, 126, 142, 154, 188, 152, 184, 194, 114, 120, 122, 118, 124, 172, 134, 156, 202,
                           204, 182, 186, 208, 170, 108, 176, 196, 200, 150, 178, 110, 116, 136, 162, 130, 146,
                           158, 198, 102, 140, 168, 104, 174]
        GIF_right_labels = [24, 32, 37, 39, 41, 45, 48, 50, 52, 56, 58, 60, 62, 77, 105, 137, 147, 179, 143, 163, 165,
                            191, 205, 125, 141, 153, 187, 151, 183, 193, 113, 119, 121, 117, 123, 171, 133, 155, 201,
                            203, 181, 185, 207, 169, 107, 175, 195, 199, 149, 177, 109, 115, 135, 161, 129, 145, 157,
                            197, 101, 139, 167, 103, 173]
        label_name = ['Accumbens Area', 'Amygdala', 'Caudate', 'Cerebellum Exterior', 'Cerebellum White Matter',
                      'Cerebral White Matter', 'Hippocampus', 'Inf Lat Vent', 'Lateral Ventricle', 'Pallidum',
                      'Putamen', 'Thalamus Proper', 'Ventral DC', 'Basal Forebrain', 'AOrG anterior orbital gyrus',
                      'LOrG lateral orbital gyrus', 'MOrG medial orbital gyrus', 'POrG posterior orbital gyrus',
                      'MFG middle frontal gyrus', 'OpIFG opercular part of the inferior frontal gyrus',
                      'OrIFG orbital part of the inferior frontal gyrus', 'SFG superior frontal gyrus',
                      'TrIFG triangular part of the inferior frontal gyrus', 'GRe gyrus rectus',
                      'MFC medial frontal cortex', 'MSFG superior frontal gyrus medial segment',
                      'SCA subcallosal area', 'MPrG precentral gyrus medial segment', 'PrG precentral gyrus',
                      'SMC supplementary motor cortex', 'CO central operculum', 'FO frontal operculum',
                      'FRP frontal pole', 'Ent entorhinal area', 'FuG fusiform gyrus', 'PHG parahippocampal gyrus',
                      'ITG inferior temporal gyrus', 'MTG middle temporal gyrus', 'STG superior temporal gyrus',
                      'TMP temporal pole', 'PP planum polare', 'PT planum temporale',
                      'TTG transverse temporal gyrus', 'PCu precuneus', 'AnG angular gyrus',
                      'PO parietal operculum', 'SMG supramarginal gyrus', 'SPL superior parietal lobule',
                      'MPoG postcentral gyrus medial segment', 'PoG postcentral gyrus', 'Calc calcarine cortex',
                      'Cun cuneus', 'LiG lingual gyrus', 'OFuG occipital fusiform gyrus',
                      'lOG inferior occipital gyrus', 'MOG middle occipital gyrus', 'OCP occipital pole',
                      'SOG superior occipital gyrus', 'ACgG anterior cingulate gyrus', 'MCgG middle cingulate gyrus',
                      'PCgG posterior cingulate gyrus', 'Alns anterior insula', 'Pins posterior insula']
        side_ = []
        name = []  #np.zeros(self.inputs.in_scan_number)
        m = min(self.inputs.ages)
        print m
        print max(self.inputs.ages)
        if max(self.inputs.ages) == m:
            age_norm = [0 for i in self.inputs.ages]
        else:
            age_norm = [(float(i) - m) / (max(self.inputs.ages) - m) for i in self.inputs.ages]

        print "age_norm = " + str(age_norm)

        print len(age_norm)
        print "len(labels) = " + str(len(labels))
        for k in range(len(self.inputs.input_meshes)):
            print "iter k : " + str(k)
            print labels[k]
            if labels[k] in GIF_left_labels:
                side_.append('Left')
                name.append(label_name[GIF_left_labels.index(labels[k])])
            if labels[k] in GIF_right_labels:
                side_.append('Right')
                name.append(label_name[GIF_right_labels.index(labels[k])])

        print name
        surf_aligned = [{'SujID': k,
                         'Filename': 'x',
                         'Age': 'x',
                         'Age_Norm': -1,  # to compute after, in the ComputeBarycenter node, should be done here.
                         'Side': side_[k],
                         'Label_GIF2': labels[k],
                         'Structure_Name': name[k]}
                        for k in range(len(self.inputs.input_meshes))]
        print surf_aligned
        if self.inputs.nbsuj_longitudinal is not -1:
            b0_surf_aligned = [{'SujID': k,
                                'Filename': 'x',
                                'Age': 'x',
                                'Age_Norm': -1,  # to compute after, in the ComputeBarycenter node
                                'Side': side_[k],
                                'Label': labels[k],
                                'Structure_Name': name[k]}
                               for k in range(self.inputs.nbsuj_longitudinal)]

        ind = 0
        ind_bo = 0
        k = 0
        i = 0
        p, bn, ext = split_filename(self.inputs.out_ageToOnsetNorm_filename)
        self.output_age_file_b0 = p + bn + '_b0' + ext
        age_file = open(self.inputs.out_ageToOnsetNorm_filename, 'w')
        age_file_b0 = open(self.output_age_file_b0, 'w')
        print "==== INPUT DATA ============"
        print "== : self.inputs.input_meshes = " + str(self.inputs.input_meshes)
        print "== : self.inputs.ages = " + str(self.inputs.ages)
        print "== : age_norm = " + str(age_norm)
        print "== :  self.inputs.subject_ids = " + str(self.inputs.subject_ids)
        print "==== INPUT DATA ============"
        while i < len(self.inputs.subject_ids):
            print " ==== DEBUG: start the loop with i: " + str(i) + " and k: " + str(k)
            print " ==== DEBUG: self.inputs.subject_ids[i] = " + self.inputs.subject_ids[i] + " self.inputs.input_meshes[k] = " + self.inputs.input_meshes[k]
            # find the index of the input_meshes containing the subj ID row[2]:
            while self.inputs.subject_ids[i] not in self.inputs.input_meshes[k]:
                k += 1
                if k > len(self.inputs.input_meshes) - 1:
                    print "=== DEBUG: end of the list. "
                    k -= 1
                    break
            print '      ' + str(i) + ' ' + str(k)
            if self.inputs.subject_ids[i] in self.inputs.input_meshes[k]:
                print "=== DEBUG: found a new match: " + self.inputs.subject_ids[i] + " and " + self.inputs.input_meshes[k]
                print '      ' + str(i) + ' ' + str(k)
                current_suj_id = self.inputs.subject_ids[i]
                if self.inputs.nbsuj_longitudinal is not -1:
                    #if current_suj_id is not
                    print "===================== DEBUG longitudinal========================================="
                    print str(i) + ", k=" + str(k) + ", ind=" + str(ind_bo)
                    if self.inputs.subject_ids[i] in self.inputs.input_meshes[k]:
                        print self.inputs.subject_ids[i] + " " + self.inputs.input_meshes[k]
                        b0_surf_aligned[ind_bo]["Filename"] = self.inputs.input_meshes[k]
                        b0_surf_aligned[ind_bo]["Age"] = self.inputs.ages[k]
                        b0_surf_aligned[ind_bo]["Age_Norm"] = age_norm[k]  # to compute after, in the ComputeBarycenter node
                        b0_surf_aligned[ind_bo]["Side"] = side_
                        b0_surf_aligned[ind_bo]["SujID"] = self.inputs.subject_ids[k]
                        age_file_b0.write(str(self.inputs.subject_ids[k]) + ' ' + self.inputs.input_meshes[k] + ' ' +
                                          str(age_norm[k]) + '\n')
                        ind_bo += 1

                # in case of longitudinal data, the followup are just after the base line, with repeated subject ID

                print '      ' + str(i) + ' ' + str(k)
                print "== DEBUG before while loop:"
                print "== DEBUG: (self.inputs.subject_ids[i] in self.inputs.input_meshes[k]) = " + str((self.inputs.subject_ids[i] in self.inputs.input_meshes[k]))
                print "== DEBUG: (current_suj_id is self.inputs.subject_ids[i]) = " + str((current_suj_id is self.inputs.subject_ids[i]))
                print "== DEBUG: while condition = " + str(((self.inputs.subject_ids[i] in self.inputs.input_meshes[k]) and (current_suj_id is self.inputs.subject_ids[i])))
                while (self.inputs.subject_ids[i] in self.inputs.input_meshes[k]) and (current_suj_id == self.inputs.subject_ids[i]):
                    print "=== DEBUG while loop: we have a match between self.inputs.subject_ids[i] " + self.inputs.subject_ids[i] + ", and current_suj_id " + current_suj_id
                    print str(i) + ' / ' + str(k)
                    print '   ' + current_suj_id + ' vs ' + self.inputs.subject_ids[i]
                    print self.inputs.subject_ids[i] + " " + self.inputs.input_meshes[k]
                    surf_aligned[ind]["Filename"] = self.inputs.input_meshes[k]
                    surf_aligned[ind]["Age"] = self.inputs.ages[k]
                    surf_aligned[ind]["Age_Norm"] = age_norm[k]  # to compute after, in the ComputeBarycenter node
                    surf_aligned[ind]["Side"] = side_[k]
                    surf_aligned[ind]["SujID"] = self.inputs.subject_ids[k]
                    age_file.write(str(self.inputs.subject_ids[k]) + ' ' + self.inputs.input_meshes[k] + ' ' +
                                          str(age_norm[k]) + '\n')
                    ind += 1
                    i += 1
                    k += 1
                    if k > len(self.inputs.input_meshes) - 1:
                        break

                if k < len(self.inputs.input_meshes) and i < len(self.inputs.subject_ids):
                    print "= = = DEBUG after while self.inputs.subject_ids[i] = " + self.inputs.subject_ids[i-1] + " in "+ self.inputs.input_meshes[k-1]
                    print "= = = DEBUG after while current_suj_id = " + current_suj_id + " =?= " + self.inputs.subject_ids[i]
                else:
                    print "= = = DEBUG after while indices out of range"

            else:  # if self.inputs.subject_ids[i] in self.inputs.input_meshes[k]:
                if k > len(self.inputs.input_meshes) - 1:
                    break
                else:
                    k -= 1
            # i = i+1
            # k = k+1
            if k > len(self.inputs.input_meshes) - 1:
                break
            if k < len(self.inputs.input_meshes) and i < len(self.inputs.subject_ids):
                print " ==== DEBUG: we end this loop with self.inputs.subject_ids[i] = " +self.inputs.subject_ids[i]+ " and self.inputs.input_meshes[k] = " + self.inputs.input_meshes[k]
                print " ==== DEBUG end loop: i: " + str(i) + " and k: " + str(k)
            else:
                print " ==== DEBUG: we end this loop with a out of range for k or i"
                print " ==== DEBUG end loop: i: " + str(i) + " and k: " + str(k)

        outputs = self._list_outputs()
        output = outputs['out_file']
        output_mat = outputs['out_file_mat']
        scipy.io.savemat(output_mat, mdict={'surf_aligned': surf_aligned})
        with open(output, 'wb') as f_obj:
            pickle.dump(surf_aligned, f_obj)

        if self.inputs.nbsuj_longitudinal is not -1:
            outputs = self._list_outputs()
            output = outputs['out_b0_file']
            output_mat = outputs['out_b0_file_mat']
            scipy.io.savemat(output_mat, mdict={'surf_aligned': b0_surf_aligned})
            with open(output, 'wb') as f_obj:
                pickle.dump(b0_surf_aligned, f_obj)

        return runtime

    @staticmethod
    def _gen_output_filename(name, ext):
        outfile = os.path.abspath(name + ext)
        return outfile


def split_list(including_id, all_id, list_data):
    # all_id correspond to list_data: list_data[0] is the data of sibject all_id[0]
    import numpy as np
    extracted_list = []
    for i in enumerate(including_id):
        indx = np.where(np.array(all_id) == i[1])
        indx = indx[0][:]
        ind = indx.tolist()[0]
        extracted_list.append(list_data[ind])

    return extracted_list

def split_list2(including_id, all_id, list_data):
    # all_id correspond to list_data: list_data[0] is the data of sibject all_id[0]
    import numpy as np
    extracted_list = []
    for i in range(len(all_id)):
        for j in range(len(including_id)):
            if all_id[i] in including_id[j]:
                extracted_list.append(list_data[i])
                break

    return extracted_list

def write_age2onset_file(age, sub_id,
                         bool_1persuj=False,
                         file_name='AgeToOnsetNorm.txt'):
    import os
    file_name = os.path.abspath(file_name)
    age_norm = [(float(i) - min(age))/(max(age)-min(age)) for i in age]
    if bool_1persuj:
        # loop on the subjects:
        p, bn, ext = split_filename(file_name)
        files_list = []
        for k in range(len(sub_id)):
            ge_file_b0 = p + bn + '_b0' + ext
            files_list.append(ge_file_b0)
            age_file = open(ge_file_b0, 'w')
            age_file.write(str(sub_id[k]) + ' ' + str(age[k]) + ' ' + str(age_norm[k]) + '\n')
        del file_name
        file_name = files_list
    else:
        age_file = open(file_name, 'w')
        for k in range(len(sub_id)):
            age_file.write(str(sub_id[k]) + ' ' + str(age[k]) + ' ' + str(age_norm[k]) + '\n')

    print file_name
    return file_name


class longitudinal_splitBaselineFollowupInputSpec(BaseInterfaceInputSpec):
  #  in_all_verticesFile = traits.List(File(exists=True, mandatory=True,
   #                                        desc=' '))
    in_all_subj_ids = traits.List(traits.String(desc=''), mandatory=True)
    in_all_ages = traits.List(traits.Float(), mandatory=True)
    in_all_meshes = traits.List(File(exists=True), mandatory=True)
    number_followup = traits.List(traits.Int(), mandatory=True)


class longitudinal_splitBaselineFollowupOutputSpec(TraitedSpec):
    #b0_vertices_files = traits.List(File(exists=True))
    b0_age2onset_norm_file = traits.File(exists=True, desc=' file containing the age to onset ' +
                                                           'of all the baseline images. ')
    b0_ages = traits.List(traits.Float())
    b0_meshes = traits.List(File(exists=True))
    subject_ids_unique = traits.List(traits.String())
    indiv_meshes = traits.List(traits.List(File()))
    indiv_age2onset_norm_file = traits.List(traits.File(exists=True))

# for longitudinal study. split the baselines from the follow up, then cluster the meshes and ages per subjects
class longitudinal_splitBaselineFollowup(BaseInterface):
    input_spec = longitudinal_splitBaselineFollowupInputSpec
    output_spec = longitudinal_splitBaselineFollowupOutputSpec

    def _run_interface(self, runtime):

        print " DEBUG self.inputs.number_followup = " + str(self.inputs.number_followup)
        m = min(self.inputs.in_all_ages)
        age_norm = [(float(i) - m) / (max(self.inputs.in_all_ages) - m) for i in self.inputs.in_all_ages]

        prev_suj = self.inputs.in_all_subj_ids[0]
        sublist_meshes = []
        self.indiv_meshes = range(len(self.inputs.number_followup))
        self.b0_meshes = range(len(self.inputs.number_followup))
        self.b0_meshes[0] = self.inputs.in_all_meshes[0]
        self.b0_ages = range(len(self.inputs.number_followup))
        self.b0_ages[0] = self.inputs.in_all_ages[0]
        #self.b0_vertices_files[0] = self.inputs.in_all_verticesFile[0]
        self.b0_age2onset_norm_file = os.path.abspath('b0_age2onset_norm_file.txt')
        file_b0 = open(self.b0_age2onset_norm_file, 'w+')
        file_b0.write(prev_suj + ' ' + self.inputs.in_all_meshes[0] + ' ' +
                      str(age_norm[0]) + '\n')
        self.indiv_age2onset_norm_file = range(len(self.inputs.number_followup))
        self.indiv_age2onset_norm_file[0] = os.path.abspath(prev_suj + '_age2onset_norm_file.txt')
        self.subject_ids_unique = range(len(self.inputs.number_followup))
        self.subject_ids_unique[0] = self.inputs.in_all_subj_ids[0]
        str_indiv_age = ''
        k = 0
        k0 = 1
        # Assuming that the meshes (therefore the input images) are given in the format [ s1 s1 s2 s2 s3 s3...]
        # if two images per subjects
        for index, suj in enumerate(self.inputs.in_all_subj_ids):
            if prev_suj == suj:
                print "== DEBUG prev_suj == suj: k = " + str(k)
                print "== DEBUG prev_suj == suj: index = " + str(index)
                print "== DEBUG prev_suj == suj: self.inputs.in_all_meshes[index] " + self.inputs.in_all_meshes[index]
                sublist_meshes.append(self.inputs.in_all_meshes[index])
                str_indiv_age = str_indiv_age + suj + ' ' + self.inputs.in_all_meshes[index] + \
                                ' ' + str(age_norm[index]) + '\n'
                if index == len(self.inputs.in_all_subj_ids) - 1:
                    print "== DEBUG  last subject: k = " + str(k)
                    print "== DEBUG  last subject: index = " + str(index)
                    self.indiv_meshes[k] = sublist_meshes
                    sublist_meshes = [self.inputs.in_all_meshes[index]]
                    print "== DEBUG last subject: "
                    file_name = suj + '_age2onset_norm_file.txt'
                    #print "== DEBUG: file_name " + file_name
                    file_indiv = open(self.indiv_age2onset_norm_file[k], 'w+')
                    print "== DEBUG: str_indiv_age " + str(str_indiv_age)
                    file_indiv.write(str_indiv_age)
                    print "== DEBUG: "
                    file_indiv.close()

            else:
                print "== DEBUG  == prev suj different from suj"
                print "== DEBUG  == suj: k = " + str(k)
                print "== DEBUG  == suj: index = " + str(index)
                print "== DEBUG: sublist_meshes " + str(sublist_meshes)
                self.indiv_meshes[k] = sublist_meshes
                print "== DEBUG: self.inputs.in_all_meshes[index] " + self.inputs.in_all_meshes[index]
                self.b0_meshes[k0] = self.inputs.in_all_meshes[index]
                print "== DEBUG: self.inputs.in_all_ages[index] " + str(self.inputs.in_all_ages[index])
                self.b0_ages[k0] = float(self.inputs.in_all_ages[index])
                print "== DEBUG: [self.inputs.in_all_meshes[index]] " + str([self.inputs.in_all_meshes[index]])
                sublist_meshes = [self.inputs.in_all_meshes[index]]
                print "== DEBUG: suj " + suj
                self.subject_ids_unique[k0] = suj
                print "== DEBUG: "
                file_b0.write(suj + ' ' + self.inputs.in_all_meshes[index] + ' ' +
                              str(age_norm[index]) + '\n')
                file_name = suj + '_age2onset_norm_file.txt'
                #print "== DEBUG: file_name " + file_name
                file_indiv = open(self.indiv_age2onset_norm_file[k], 'w+')
                print "== DEBUG: str_indiv_age " + str(str_indiv_age)
                file_indiv.write(str_indiv_age)
                print "== DEBUG: "
                file_indiv.close()
                self.indiv_age2onset_norm_file[k0] = os.path.abspath(suj + '_age2onset_norm_file.txt')
                str_indiv_age = suj + ' ' + self.inputs.in_all_meshes[index] + ' ' + \
                                str(age_norm[index]) + '\n'
                #file_b0.write(str_indiv_age)
                k += 1
                k0 += 1
                prev_suj = suj

        file_b0.close()
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['b0_ages'] = self.b0_ages
        outputs['b0_age2onset_norm_file'] = self.b0_age2onset_norm_file
        outputs['b0_meshes'] = self.b0_meshes
        outputs['subject_ids_unique'] = self.subject_ids_unique
        outputs['indiv_meshes'] = self.indiv_meshes
        outputs['indiv_age2onset_norm_file'] = self.indiv_age2onset_norm_file
        return outputs


class reorder_listsInputSpec(BaseInterfaceInputSpec):
    in_list_a = traits.List(desc='list in the form [a1 a2 a3] [b1 b2 b3] to be reorder in [[a1 b1] [a2 b2] [a3 b3]]')
    in_list_b = traits.List(desc='list in the form [a1 a2 a3] [b1 b2 b3] to be reorder in [[a1 b1] [a2 b2] [a3 b3]]')


class reorder_listsOutputSpec(TraitedSpec):
    sorted_list = traits.List(traits.List())


class reorder_lists(BaseInterface):
    input_spec = reorder_listsInputSpec
    output_spec = reorder_listsOutputSpec

    def _run_interface(self, runtime):
        self.sorted_list = []
        list_a = self.inputs.in_list_a
        list_b = self.inputs.in_list_b
        print list_a
        if len(list_a) != len(list_b):
            raise ValueError("list a and list b have different size.")

        for i in range(len(list_a)):
            sub = []
            sub.append(list_a[i])
            sub.append(list_b[i])
            print sub
            self.sorted_list.append(sub)

        print self.sorted_list
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['sorted_list'] = self.sorted_list

        return outputs


class reorder_lists2InputSpec(BaseInterfaceInputSpec):
    in_list = traits.List(desc='list with even length to be rearrange (2 lists merged). ' +
                               '[a a a b b b] --> [[a b] [a b] [a b]]')


class reorder_lists2OutputSpec(TraitedSpec):
    sorted_list = traits.List(traits.List())


class reorder_lists2(BaseInterface):
    input_spec = reorder_lists2InputSpec
    output_spec = reorder_lists2OutputSpec

    def _run_interface(self, runtime):
        size_list = len(self.inputs.in_list)
        if size_list % 2 == 0:
            print "The list have a even size"
        else:
            print " reorder_lists2: something went wrong, the list should have an even size."

        self.sorted_list = []
        list_a = [self.inputs.in_list[i] for i in range(size_list / 2)]
        list_b = [self.inputs.in_list[i] for i in range(size_list / 2, size_list)]
        if len(list_a) != len(list_b):
            raise ValueError("list a and list b have different size.")

        for i in range(len(list_a)):
            sub = []
            sub.append(list_a[i])
            sub.append(list_b[i])
            self.sorted_list.append(sub)

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['sorted_list'] = self.sorted_list

        return outputs


class regional_analysisInputSpec(BaseInterfaceInputSpec):
    in_residual_v0_transported = traits.List(traits.File(exists=True), mandatory=True)
    in_cp0_file = traits.File(exists=True, desc='txt file, containing the positions of the control points, ' +
                                                'where the residual V0 are transported, and then defined.')
    in_k = traits.Int(10, usedefault=True, desc='number of cluster, for k-means')

class regional_analysisOutputSpec(TraitedSpec):
    residual_v0_transported_clustered = traits.List(traits.File())


class regional_analysis(BaseInterface):
    input_spec = regional_analysisInputSpec
    output_spec = regional_analysisOutputSpec

    def _run_interface(self, runtime):
        cp_0=[]
        cp_0_f = open(self.inputs.in_cp0_file)
        for i, line in enumerate(cp_0_f):
            print (i, line)
            cp_0[i] = line.split()
        cp_0_f.close()

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        return outputs


class infoExtractionCSVInputSpec(BaseInterfaceInputSpec):
    in_csv_file = File(exists=True)
    in_subID = traits.List()
    in_var_tobe_extracted = traits.List(desc='list of colunm names: TIV, Gender,...')

class infoExtractionCSVOutputSpec(TraitedSpec):
    residual_v0_transported_clustered = traits.List(traits.File())


class infoExtractionCSV(BaseInterface):
    input_spec = infoExtractionCSVInputSpec
    output_spec = infoExtractionCSVOutputSpec

    def _run_interface(self, runtime):

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        return outputs
