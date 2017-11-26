Node: spatio_temporal_regression_preprocessing (age2onset_file_age (utility)
============================================================================

 Hierarchy : spatio_temporal_analysis.spatio_temporal_regression_preprocessing.age2onset_file_age
 Exec ID : age2onset_file_age

Original Inputs
---------------

* age : [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]
* bool_1persuj : False
* function_str : def write_age2onset_file(age, sub_id,
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

* ignore_exception : False
* sub_id : ['S1', 'S1', 'S2', 'S2', 'S3', 'S3', 'S4', 'S4']

Execution Inputs
----------------

* age : [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]
* bool_1persuj : False
* function_str : def write_age2onset_file(age, sub_id,
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

* ignore_exception : False
* sub_id : ['S1', 'S1', 'S2', 'S2', 'S3', 'S3', 'S4', 'S4']

Execution Outputs
-----------------

* file_name : /Users/clairec/Codes/Source/NiftyPipe/test_symmetric_04Mars/spatio_temporal_analysis/spatio_temporal_regression_preprocessing/age2onset_file_age/AgeToOnsetNorm.txt

Runtime info
------------

* duration : 0.06925
* hostname : Claires-MBP-2

Environment
~~~~~~~~~~~

* Apple_PubSub_Socket_Render : /private/tmp/com.apple.launchd.H9swusR8RZ/Render
* CLICOLOR : 1
* DISPLAY : /private/tmp/com.apple.launchd.NRGk5KhTX9/org.macosforge.xquartz:0
* DYLD_FALLBACK_LIBRARY_PATH : /Users/clairec/Codes/Install/afni_install/macosx_10.7_Intel_64:
* DYLD_LIBRARY_PATH : /Users/clairec/Codes/Source/armadillo-6.400.3:/Users/clairec/Codes/Install/VTK-6.1.0_install/lib:/Users/clairec/Codes/Install/InsightToolkit-4.8.0_install/lib::/Users/clairec/Codes/Build/NiftyReg_build/reg-lib
* FSLDIR : /Applications/fsl
* FSLGECUDAQ : cuda.q
* FSLLOCKDIR : 
* FSLMACHINELIST : 
* FSLMULTIFILEQUIT : TRUE
* FSLOUTPUTTYPE : NIFTI_GZ
* FSLREMOTECALL : 
* FSLTCLSH : /Applications/fsl/bin/fsltclsh
* FSLWISH : /Applications/fsl/bin/fslwish
* HOME : /Users/clairec
* LC_CTYPE : en_US.UTF-8
* LOGNAME : clairec
* LSCOLORS : ExFxBxDxCxegedabagacad
* OMP_NUM_THREADS : 1
* PATH : /opt/local/bin:/opt/local/sbin:/opt/local/bin/pkg-config:/Users/clairec/Codes/Install/afni_install/macosx_10.7_Intel_64:/Applications/fsl/bin:/opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin:/Applications/fsl/bin:/Applications/MATLAB_R2015a.app/bin:/Users/clairec/Codes/Build/Deformetrica_dev_build:/Users/clairec/Codes/Source/deformetrica/deformetrica/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/X11/bin:/Library/TeX/texbin:/Users/clairec/Codes/Install/NiftyReg/bin:/Users/clairec/Codes/Install/NiftySeg/bin:/Applications/niftk-15.04.0/NiftyView.app/Contents/MacOS
* PS1 : \[\033[36m\]\u\[\033[m\]@\[\033[1;34m\]\h:\[\033[33;1m\]\w\[\033[m\]$ 
* PWD : /Users/clairec/Codes/Source/NiftyPipe
* PYTHONPATH : /Users/clairec/Codes/NiftyPipe/interfaces::.:/Users/clairec/NifTK-src/Code/nipype-workflows/diffusion:/Users/clairec/NifTK-src/Code/nipype-workflows/registration:/Users/clairec/NifTK-src/Code/nipype-workflows/seg-gif:/Users/clairec/NifTK-src/Code/nipype-workflows/utils
* SHAPE_PATH : /Users/clairec/Codes/Code2
* SHELL : /bin/bash
* SHLVL : 1
* SSH_AUTH_SOCK : /private/tmp/com.apple.launchd.hcAUuBPpWz/Listeners
* TERM : xterm-256color
* TMPDIR : /var/folders/nq/0w7jryf96r36zffyg3n6yk4r0000gn/T/
* USER : clairec
* XPC_FLAGS : 0x0
* XPC_SERVICE_NAME : 0
* _ : /opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin/perform_spatiotemporal_shape_symmetric.py
* __CF_USER_TEXT_ENCODING : 0x1F5:0x0:0x0

