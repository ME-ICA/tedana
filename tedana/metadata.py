"""Ingestion of BIDS acquisition metadata (one JSON per echo).

Metadata files are supplied explicitly by the user (imaging files are
preprocessing derivatives; the JSONs come from the raw BIDS dataset), so this
module never discovers files -- it parses exactly the paths it is given.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

LGR = logging.getLogger("GENERAL")

_AXIS_FROM_LETTER = {"i": 0, "j": 1, "k": 2}


@dataclass
class AcquisitionMetadata:
    """Parsed + derived acquisition parameters used by metadata-aware metrics."""

    tes: List[float]
    n_slices: Optional[int] = None
    slice_axis: Optional[int] = None
    mb_factor: Optional[int] = None
    pe_direction: Optional[str] = None
    in_plane_accel: Optional[int] = None
    repetition_time: Optional[float] = None
    source: dict = field(default_factory=dict)


def _infer_mb_from_slice_timing(slice_timing, tol=1e-4):
    """Multiband factor = slices sharing each acquisition time (n_slices / n_time_groups)."""
    times = np.asarray(slice_timing, dtype=float)
    n = times.size
    groups = []
    for t in times:
        if not any(abs(t - g) <= tol for g in groups):
            groups.append(t)
    n_groups = len(groups)
    if n_groups == 0 or n % n_groups != 0:
        return None
    return n // n_groups


def parse_acquisition_metadata(metadata_paths: List[str], n_echos: int) -> AcquisitionMetadata:
    """Parse one BIDS JSON per echo (in ``data`` order) into AcquisitionMetadata."""
    if len(metadata_paths) != n_echos:
        raise ValueError(
            f"Number of metadata files ({len(metadata_paths)}) does not match "
            f"number of echoes ({n_echos})."
        )

    per_echo = []
    for path in metadata_paths:
        with open(path) as fobj:
            per_echo.append(json.load(fobj))

    tes = []
    for path, meta in zip(metadata_paths, per_echo):
        if "EchoTime" not in meta:
            raise ValueError(f"Metadata file {path} is missing required field 'EchoTime'.")
        tes.append(float(meta["EchoTime"]))  # BIDS seconds; tedana works in seconds

    source = {}

    def shared(key):
        values = [m[key] for m in per_echo if key in m]
        if not values:
            return None
        first = values[0]
        if any(v != first for v in values[1:]):
            LGR.warning(f"Metadata field '{key}' disagrees across echoes; using first value.")
        source[key] = "metadata"
        return first

    slice_timing = shared("SliceTiming")
    mb = shared("MultibandAccelerationFactor")
    in_plane = shared("ParallelReductionFactorInPlane")
    pe_direction = shared("PhaseEncodingDirection")
    slice_enc = shared("SliceEncodingDirection")
    tr = shared("RepetitionTime")

    n_slices = len(slice_timing) if slice_timing is not None else None

    slice_axis = None
    if slice_enc is not None:
        slice_axis = _AXIS_FROM_LETTER.get(slice_enc.rstrip("-"))
        source["slice_axis"] = "SliceEncodingDirection"
    elif slice_timing is not None:
        slice_axis = 2  # through-plane k by convention
        source["slice_axis"] = "default_k"

    mb_factor = None
    if mb is not None:
        mb_factor = int(mb)
        source["mb_factor"] = "MultibandAccelerationFactor"
    elif slice_timing is not None:
        mb_factor = _infer_mb_from_slice_timing(slice_timing)
        if mb_factor is not None:
            source["mb_factor"] = "inferred_from_SliceTiming"

    return AcquisitionMetadata(
        tes=tes,
        n_slices=n_slices,
        slice_axis=slice_axis,
        mb_factor=mb_factor,
        pe_direction=pe_direction,
        in_plane_accel=int(in_plane) if in_plane is not None else None,
        repetition_time=float(tr) if tr is not None else None,
        source=source,
    )


def resolve_echo_times(*, tes, metadata, n_echos):
    """Enforce the tes/metadata exclusivity and return (tes_list, AcquisitionMetadata|None).

    Exactly one of ``tes`` or ``metadata`` must be provided. With ``metadata``,
    echo times are read from the JSONs' ``EchoTime`` fields.
    """
    if (tes is None) == (metadata is None):
        raise ValueError("Provide exactly one of `tes` or `metadata` (not both, not neither).")
    if metadata is not None:
        acq = parse_acquisition_metadata(metadata, n_echos=n_echos)
        return list(acq.tes), acq
    return list(tes), None
