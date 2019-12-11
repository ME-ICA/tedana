# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:

# Overrides submodules with their functions.
from .tedana import tedana_workflow
from .t2smap import t2smap_workflow


__all__ = ['tedana_workflow', 't2smap_workflow']
