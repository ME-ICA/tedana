import json

import pytest

from tedana.metadata import parse_acquisition_metadata


def _write(tmp_path, name, d):
    p = tmp_path / name
    p.write_text(json.dumps(d))
    return str(p)


def _echo_jsons(tmp_path, tes, **shared):
    return [
        _write(tmp_path, f"echo{i}.json", {"EchoTime": te, **shared}) for i, te in enumerate(tes)
    ]


def test_parse_assembles_tes_in_order_no_conversion(tmp_path):
    paths = _echo_jsons(tmp_path, [0.015, 0.039, 0.063])
    md = parse_acquisition_metadata(paths, n_echos=3)
    assert md.tes == [0.015, 0.039, 0.063]  # seconds, no conversion


def test_parse_derives_slice_axis_and_mb_factor(tmp_path):
    paths = _echo_jsons(
        tmp_path,
        [0.015, 0.039],
        MultibandAccelerationFactor=3,
        SliceEncodingDirection="k",
        SliceTiming=[0.0, 0.1, 0.2] * 4,  # 12 slices
        ParallelReductionFactorInPlane=2,
        PhaseEncodingDirection="j-",
    )
    md = parse_acquisition_metadata(paths, n_echos=2)
    assert md.slice_axis == 2
    assert md.mb_factor == 3
    assert md.n_slices == 12
    assert md.in_plane_accel == 2
    assert md.pe_direction == "j-"


def test_parse_infers_mb_from_slice_timing(tmp_path):
    # 12 slices, 4 distinct acquisition times of 3 slices each -> MB 3
    timing = [0.0, 0.1, 0.2, 0.3] * 3
    paths = _echo_jsons(tmp_path, [0.015, 0.039], SliceTiming=timing)
    md = parse_acquisition_metadata(paths, n_echos=2)
    assert md.mb_factor == 3
    assert md.slice_axis == 2  # defaults to k when only SliceTiming present


def test_parse_length_mismatch_errors(tmp_path):
    paths = _echo_jsons(tmp_path, [0.015, 0.039])
    with pytest.raises(ValueError, match="does not match"):
        parse_acquisition_metadata(paths, n_echos=3)


def test_parse_missing_echotime_errors(tmp_path):
    p = _write(tmp_path, "echo0.json", {"SliceTiming": [0.0, 0.1]})
    with pytest.raises(ValueError, match="EchoTime"):
        parse_acquisition_metadata([p], n_echos=1)


def test_parse_missing_optional_fields_fall_back(tmp_path):
    paths = _echo_jsons(tmp_path, [0.015, 0.039])  # only EchoTime
    md = parse_acquisition_metadata(paths, n_echos=2)
    assert md.slice_axis is None
    assert md.mb_factor is None
    assert md.n_slices is None


def test_infer_mb_non_divisible_returns_none():
    from tedana.metadata import _infer_mb_from_slice_timing

    # 5 slices, 3 distinct acquisition times -> not divisible -> None
    assert _infer_mb_from_slice_timing([0.0, 0.1, 0.2, 0.0, 0.1]) is None


def test_parse_warns_on_cross_echo_disagreement(tmp_path, caplog):
    import logging

    p0 = _write(tmp_path, "e0.json", {"EchoTime": 0.015, "RepetitionTime": 1.5})
    p1 = _write(tmp_path, "e1.json", {"EchoTime": 0.039, "RepetitionTime": 2.0})
    with caplog.at_level(logging.WARNING, logger="GENERAL"):
        md = parse_acquisition_metadata([p0, p1], n_echos=2)
    assert md.repetition_time == 1.5  # first value used
    assert any("RepetitionTime" in r.message for r in caplog.records)


def test_resolve_tes_and_metadata_exclusivity():
    from tedana.metadata import resolve_echo_times

    # neither -> error
    with pytest.raises(ValueError, match="exactly one"):
        resolve_echo_times(tes=None, metadata=None, n_echos=2)
    # both -> error
    with pytest.raises(ValueError, match="exactly one"):
        resolve_echo_times(tes=[0.015, 0.039], metadata=["a.json"], n_echos=2)


def test_resolve_tes_from_metadata(tmp_path):
    from tedana.metadata import resolve_echo_times

    paths = _echo_jsons(
        tmp_path,
        [0.015, 0.039],
        MultibandAccelerationFactor=3,
        SliceTiming=[0.0, 0.1, 0.2] * 4,
        SliceEncodingDirection="k",
    )
    tes, md = resolve_echo_times(tes=None, metadata=paths, n_echos=2)
    assert tes == [0.015, 0.039]
    assert md is not None and md.mb_factor == 3


def test_resolve_tes_from_tes_no_metadata():
    from tedana.metadata import resolve_echo_times

    tes, md = resolve_echo_times(tes=[0.015, 0.039], metadata=None, n_echos=2)
    assert tes == [0.015, 0.039]
    assert md is None
