# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..utils import SplitAndSelect


def test_SplitAndSelect_inputs():
    input_map = dict(ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(mandatory=True,
    ),
    in_id=dict(mandatory=False,
    usedefault=True,
    ),
    )
    inputs = SplitAndSelect.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_SplitAndSelect_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = SplitAndSelect.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
