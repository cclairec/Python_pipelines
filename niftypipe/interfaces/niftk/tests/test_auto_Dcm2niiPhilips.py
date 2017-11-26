# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..io import Dcm2niiPhilips


def test_Dcm2niiPhilips_inputs():
    input_map = dict(anonymize=dict(argstr='-a',
    usedefault=True,
    ),
    args=dict(argstr='%s',
    ),
    collapse_folders=dict(argstr='-c',
    usedefault=True,
    ),
    config_file=dict(argstr='-b %s',
    genfile=True,
    ),
    convert_all_pars=dict(argstr='-v',
    usedefault=True,
    ),
    date_in_filename=dict(argstr='-d',
    usedefault=True,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    events_in_filename=dict(argstr='-e',
    usedefault=True,
    ),
    gzip_output=dict(argstr='-g',
    usedefault=True,
    ),
    id_in_filename=dict(argstr='-i',
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    nii_output=dict(argstr='-n',
    usedefault=True,
    ),
    output_dir=dict(argstr='-o %s',
    genfile=True,
    ),
    philips_precise=dict(argstr='',
    usedefault=True,
    ),
    protocol_in_filename=dict(argstr='-p',
    usedefault=True,
    ),
    reorient=dict(argstr='-r',
    ),
    reorient_and_crop=dict(argstr='-x',
    usedefault=True,
    ),
    source_dir=dict(argstr='%s',
    mandatory=True,
    position=-1,
    xor=['source_names'],
    ),
    source_in_filename=dict(argstr='-f',
    usedefault=True,
    ),
    source_names=dict(argstr='%s',
    copyfile=False,
    mandatory=True,
    position=-1,
    xor=['source_dir'],
    ),
    spm_analyze=dict(argstr='-s',
    xor=['nii_output'],
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = Dcm2niiPhilips.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_Dcm2niiPhilips_outputs():
    output_map = dict(b0map=dict(),
    bvals=dict(),
    bvecs=dict(),
    converted_files=dict(),
    dirsense=dict(),
    flair=dict(),
    mtr=dict(),
    outputdir=dict(),
    pdt2=dict(),
    philips_dwi=dict(),
    psir=dict(),
    reoriented_and_cropped_files=dict(),
    reoriented_files=dict(),
    t12d=dict(),
    t13d=dict(),
    )
    outputs = Dcm2niiPhilips.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
