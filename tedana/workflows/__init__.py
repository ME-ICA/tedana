# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""Command line interfaces and workflows."""
from tedana.workflows.ica_reclassify import ica_reclassify_workflow
from tedana.workflows.t2smap import t2smap_workflow
from tedana.workflows.tedana import tedana_workflow

__all__ = ["tedana_workflow", "t2smap_workflow", "ica_reclassify_workflow"]
