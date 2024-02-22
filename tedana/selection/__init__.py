"""TEDANA selection methods."""

from tedana.selection.tedica import automatic_selection
from tedana.selection.tedpca import kundu_tedpca

__all__ = ["kundu_tedpca", "automatic_selection"]
