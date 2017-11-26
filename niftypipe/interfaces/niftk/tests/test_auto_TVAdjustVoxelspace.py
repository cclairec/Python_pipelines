# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..dtitk import TVAdjustVoxelspace


def test_TVAdjustVoxelspace_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='-in %s',
    mandatory=True,
    ),
    orig_val_x=dict(argstr='-origin %f',
    mandatory=True,
    position=1,
    ),
    orig_val_y=dict(argstr='%f',
    mandatory=True,
    position=2,
    ),
    orig_val_z=dict(argstr='%f',
    mandatory=True,
    position=3,
    ),
    out_file=dict(argstr='-out %s',
    genfile=True,
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = TVAdjustVoxelspace.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_TVAdjustVoxelspace_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = TVAdjustVoxelspace.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value