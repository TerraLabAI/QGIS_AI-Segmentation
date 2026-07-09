"""Mixin package backing AISegmentationPlugin (ai_segmentation_plugin.py).

One concern per module so parallel edits never touch the same file. Method
names are unique across the whole class: never define the same method in
two mixins (MRO would silently pick one). Shared module-level constants and
free functions live in shared.py.
"""
