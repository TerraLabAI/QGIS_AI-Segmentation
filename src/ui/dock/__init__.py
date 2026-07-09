"""Mixin package backing AISegmentationDockWidget
(ai_segmentation_dockwidget.py).

One concern per module so parallel edits never collide. Method names are
unique across the class: never define the same method in two mixins (MRO
would silently pick one). Qt signals stay in the assembly class body.
Styles/colors live in styles.py; small child widgets in widgets.py.
"""
