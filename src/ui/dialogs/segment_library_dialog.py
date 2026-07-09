"""Compatibility shim: the Segment library now lives in the
``segment_library`` package (dialog, cards, detail popups, workers split per
concern). Import ``SegmentLibraryDialog`` from here as before."""
from .segment_library import SegmentLibraryDialog

__all__ = ["SegmentLibraryDialog"]
