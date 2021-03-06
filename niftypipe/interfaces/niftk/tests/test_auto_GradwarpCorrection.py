# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..distortion import GradwarpCorrection


def test_GradwarpCorrection_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    coeff_file=dict(argstr='-c %s',
    mandatory=True,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='-i %s',
    mandatory=True,
    ),
    offset_x=dict(argstr='-off_x %f',
    ),
    offset_y=dict(argstr='-off_y %f',
    ),
    offset_z=dict(argstr='-off_z %f',
    ),
    out_file=dict(argstr='-o %s',
    name_source=['in_file'],
    name_template='%s_unwarp_field',
    ),
    output_type=dict(),
    radius=dict(argstr='-r %f',
    ),
    scanner_type=dict(argstr='-t %s',
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = GradwarpCorrection.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_GradwarpCorrection_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = GradwarpCorrection.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
