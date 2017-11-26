# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..dtitk import AffineSymTensor3DVolume


def test_AffineSymTensor3DVolume_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    flo_file=dict(argstr='-in %s',
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_trans=dict(argstr='-trans %s',
    exist=True,
    ),
    out_file=dict(argstr='-out %s',
    genfile=True,
    ),
    ref_file=dict(argstr='-target %s',
    ),
    sm_option_val=dict(argstr='-interp %s',
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = AffineSymTensor3DVolume.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_AffineSymTensor3DVolume_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = AffineSymTensor3DVolume.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
