# SPDX-FileCopyrightText: 2026 TerraLab <yvann.barbot@terra-lab.ai>
# SPDX-License-Identifier: GPL-2.0-or-later
# ruff: noqa: E501
"""    Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as
    part of Feather (MIT). All other copyright (c) for Lucide are held by
    Lucide Contributors 2022.

    Permission to use, copy, modify, and/or distribute this software for
    any purpose with or without fee is hereby granted, provided that the
    above copyright notice and this permission notice appear in all copies."""













from __future__ import annotations

from qgis.PyQt.QtCore import QByteArray, QRectF
from qgis.PyQt.QtGui import QColor, QPainter

PREFIX = "lu."


_SPARKLE = '<path fill="{colour}" stroke="none" d="M12 2l2.4 7.2L22 12l-7.6 2.8L12 22l-2.4-7.2L2 12l7.6-2.8z"/>'

SHAPES = {
    "undo-2": '<path d="M9 14 4 9l5-5"/> <path d="M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5a5.5 5.5 0 0 1-5.5 5.5H11"/>',
    "redo-2": '<path d="m15 14 5-5-5-5"/> <path d="M20 9H9.5A5.5 5.5 0 0 0 4 14.5A5.5 5.5 0 0 0 9.5 20H13"/>',
    "history": (
        '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/> <path d="M3 3v5h5"/> <path'
        ' d="M12 7v5l4 2"/>'
    ),
    "arrow-up": '<path d="m5 12 7-7 7 7"/> <path d="M12 19V5"/>',
    "book-open": (
        '<path d="M12 7v14"/> <path d="M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 '
        '0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z"/>'
    ),
    "braces": (
        '<path d="M8 3H7a2 2 0 0 0-2 2v5a2 2 0 0 1-2 2 2 2 0 0 1 2 2v5c0 1.1.9 2 2 2h1"/> <path '
        'd="M16 21h1a2 2 0 0 0 2-2v-5c0-1.1.9-2 2-2a2 2 0 0 1-2-2V5a2 2 0 0 0-2-2h-1"/>'
    ),
    "calculator": (
        '<rect width="16" height="20" x="4" y="2" rx="2"/> <line x1="8" x2="16" y1="6" y2="6"/> '
        '<line x1="16" x2="16" y1="14" y2="18"/> <path d="M16 10h.01"/> <path d="M12 10h.01"/> '
        '<path d="M8 10h.01"/> <path d="M12 14h.01"/> <path d="M8 14h.01"/> <path d="M12 '
        '18h.01"/> <path d="M8 18h.01"/>'
    ),
    "camera": (
        '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 '
        '0-2-2h-3l-2.5-3z"/> <circle cx="12" cy="13" r="3"/>'
    ),
    "chart-column": (
        '<path d="M3 3v16a2 2 0 0 0 2 2h16"/> <path d="M18 17V9"/> <path d="M13 17V5"/> <path '
        'd="M8 17v-3"/>'
    ),
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "cog": (
        '<path d="M12 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16Z"/> <path d="M12 14a2 2 0 1 0 0-4 2 2 0 0 '
        '0 0 4Z"/> <path d="M12 2v2"/> <path d="M12 22v-2"/> <path d="m17 20.66-1-1.73"/> <path '
        'd="M11 10.27 7 3.34"/> <path d="m20.66 17-1.73-1"/> <path d="m3.34 7 1.73 1"/> <path '
        'd="M14 12h8"/> <path d="M2 12h2"/> <path d="m20.66 7-1.73 1"/> <path d="m3.34 17 '
        '1.73-1"/> <path d="m17 3.34-1 1.73"/> <path d="m11 13.73-4 6.93"/>'
    ),
    "combine": (
        '<path d="M10 18H5a3 3 0 0 1-3-3v-1"/> <path d="M14 2a2 2 0 0 1 2 2v4a2 2 0 0 1-2 2"/> '
        '<path d="M20 2a2 2 0 0 1 2 2v4a2 2 0 0 1-2 2"/> <path d="m7 21 3-3-3-3"/> <rect x="14" '
        'y="14" width="8" height="8" rx="2"/> <rect x="2" y="2" width="8" height="8" rx="2"/>'
    ),
    "corner-down-left": '<polyline points="9 10 4 15 9 20"/> <path d="M20 4v7a4 4 0 0 1-4 4H4"/>',
    "crosshair": (
        '<circle cx="12" cy="12" r="10"/> <line x1="22" x2="18" y1="12" y2="12"/> <line x1="6" '
        'x2="2" y1="12" y2="12"/> <line x1="12" x2="12" y1="6" y2="2"/> <line x1="12" x2="12" '
        'y1="22" y2="18"/>'
    ),
    "database": (
        '<ellipse cx="12" cy="5" rx="9" ry="3"/> <path d="M3 5V19A9 3 0 0 0 21 19V5"/> <path '
        'd="M3 12A9 3 0 0 0 21 12"/>'
    ),
    "eye": (
        '<path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 '
        '10.75 10.75 0 0 1-19.876 0"/> <circle cx="12" cy="12" r="3"/>'
    ),
    "file-down": (
        '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/> <path d="M14 '
        '2v4a2 2 0 0 0 2 2h4"/> <path d="M12 18v-6"/> <path d="m9 15 3 3 3-3"/>'
    ),
    "file-text": (
        '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/> <path d="M14 '
        '2v4a2 2 0 0 0 2 2h4"/> <path d="M10 9H8"/> <path d="M16 13H8"/> <path d="M16 17H8"/>'
    ),
    "filter": '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    "folder-open": (
        '<path d="m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 '
        '1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2'
        ' 0 0 1 2 2v2"/>'
    ),
    "globe": (
        '<circle cx="12" cy="12" r="10"/> <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 '
        '0-20"/> <path d="M2 12h20"/>'
    ),
    "grid-3x3": (
        '<rect width="18" height="18" x="3" y="3" rx="2"/> <path d="M3 9h18"/> <path d="M3 '
        '15h18"/> <path d="M9 3v18"/> <path d="M15 3v18"/>'
    ),
    "image": (
        '<rect width="18" height="18" x="3" y="3" rx="2" ry="2"/> <circle cx="9" cy="9" r="2"/> '
        '<path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>'
    ),
    "layers": (
        '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66'
        ' 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/> <path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 '
        '17.65"/> <path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>'
    ),
    "layout-template": (
        '<rect width="18" height="7" x="3" y="3" rx="1"/> <rect width="9" height="7" x="3" y="14"'
        ' rx="1"/> <rect width="5" height="7" x="16" y="14" rx="1"/>'
    ),
    "link": (
        '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/> <path d="M14 '
        '11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>'
    ),
    "list-checks": (
        '<path d="m3 17 2 2 4-4"/> <path d="m3 7 2 2 4-4"/> <path d="M13 6h8"/> <path d="M13 '
        '12h8"/> <path d="M13 18h8"/>'
    ),
    "locate-fixed": (
        '<line x1="2" x2="5" y1="12" y2="12"/> <line x1="19" x2="22" y1="12" y2="12"/> <line '
        'x1="12" x2="12" y1="2" y2="5"/> <line x1="12" x2="12" y1="19" y2="22"/> <circle cx="12" '
        'cy="12" r="7"/> <circle cx="12" cy="12" r="3"/>'
    ),
    "map-pin": (
        '<path d="M20 10c0 4.993-5.539 10.193-7.399 11.799a1 1 0 0 1-1.202 0C9.539 20.193 4 '
        '14.993 4 10a8 8 0 0 1 16 0"/> <circle cx="12" cy="10" r="3"/>'
    ),
    "message-circle-question": (
        '<path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"/> <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 '
        '3"/> <path d="M12 17h.01"/>'
    ),
    "minus": '<path d="M5 12h14"/>',
    "mountain": '<path d="m8 3 4 8 5-5 5 15H2L8 3z"/>',
    "palette": (
        '<circle cx="13.5" cy="6.5" r=".5" fill="currentColor"/> <circle cx="17.5" cy="10.5" '
        'r=".5" fill="currentColor"/> <circle cx="8.5" cy="7.5" r=".5" fill="currentColor"/> '
        '<circle cx="6.5" cy="12.5" r=".5" fill="currentColor"/> <path d="M12 2C6.5 2 2 6.5 2 '
        '12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 '
        '0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 '
        '1.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.554C21.965 6.012 17.461 2 12 2z"/>'
    ),
    "pencil": (
        '<path d="M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 '
        '4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z"/> <path d="m15 5 4 4"/>'
    ),
    "puzzle": (
        '<path d="M15.39 4.39a1 1 0 0 0 1.68-.474 2.5 2.5 0 1 1 3.014 3.015 1 1 0 0 0-.474 '
        '1.68l1.683 1.682a2.414 2.414 0 0 1 0 3.414L19.61 15.39a1 1 0 0 1-1.68-.474 2.5 2.5 0 1 '
        '0-3.014 3.015 1 1 0 0 1 .474 1.68l-1.683 1.682a2.414 2.414 0 0 1-3.414 0L8.61 19.61a1 1 '
        '0 0 0-1.68.474 2.5 2.5 0 1 1-3.014-3.015 1 1 0 0 0 .474-1.68l-1.683-1.682a2.414 2.414 0 '
        '0 1 0-3.414L4.39 8.61a1 1 0 0 1 1.68.474 2.5 2.5 0 1 0 3.014-3.015 1 1 0 0 '
        '1-.474-1.68l1.683-1.682a2.414 2.414 0 0 1 3.414 0z"/>'
    ),
    "refresh-cw": (
        '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/> <path d="M21 3v5h-5"/> '
        '<path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/> <path d="M8 16H3v5"/>'
    ),
    "route": (
        '<circle cx="6" cy="19" r="3"/> <path d="M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 '
        '0-7H15"/> <circle cx="18" cy="5" r="3"/>'
    ),
    "ruler": (
        '<path d="M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.41 2.41 0'
        ' 0 1 0-3.4l2.6-2.6a2.41 2.41 0 0 1 3.4 0Z"/> <path d="m14.5 12.5 2-2"/> <path d="m11.5 '
        '9.5 2-2"/> <path d="m8.5 6.5 2-2"/> <path d="m17.5 15.5 2-2"/>'
    ),
    "scan": (
        '<path d="M3 7V5a2 2 0 0 1 2-2h2"/> <path d="M17 3h2a2 2 0 0 1 2 2v2"/> <path d="M21 '
        '17v2a2 2 0 0 1-2 2h-2"/> <path d="M7 21H5a2 2 0 0 1-2-2v-2"/>'
    ),
    "search": '<circle cx="11" cy="11" r="8"/> <path d="m21 21-4.3-4.3"/>',
    "square-dashed-mouse-pointer": (
        '<path d="M12.034 12.681a.498.498 0 0 1 .647-.647l9 3.5a.5.5 0 0 1-.033.943l-3.444 '
        '1.068a1 1 0 0 0-.66.66l-1.067 3.443a.5.5 0 0 1-.943.033z"/> <path d="M5 3a2 2 0 0 0-2 '
        '2"/> <path d="M19 3a2 2 0 0 1 2 2"/> <path d="M5 21a2 2 0 0 1-2-2"/> <path d="M9 3h1"/> '
        '<path d="M9 21h2"/> <path d="M14 3h1"/> <path d="M3 9v1"/> <path d="M21 9v2"/> <path '
        'd="M3 14v1"/>'
    ),
    "square-terminal": (
        '<path d="m7 11 2-2-2-2"/> <path d="M11 13h4"/> <rect width="18" height="18" x="3" y="3" '
        'rx="2" ry="2"/>'
    ),
    "table-2": (
        '<path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 '
        '21H5a2 2 0 0 1-2-2V9m0 0h18"/>'
    ),
    "tag": (
        '<path d="M12.586 2.586A2 2 0 0 0 11.172 2H4a2 2 0 0 0-2 2v7.172a2 2 0 0 0 .586 '
        '1.414l8.704 8.704a2.426 2.426 0 0 0 3.42 0l6.58-6.58a2.426 2.426 0 0 0 0-3.42z"/> '
        '<circle cx="7.5" cy="7.5" r=".5" fill="currentColor"/>'
    ),
    "trash-2": (
        '<path d="M3 6h18"/> <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/> <path d="M8 '
        '6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/> <line x1="10" x2="10" y1="11" y2="17"/> <line x1="14"'
        ' x2="14" y1="11" y2="17"/>'
    ),
    "triangle-alert": (
        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/> '
        '<path d="M12 9v4"/> <path d="M12 17h.01"/>'
    ),
    "wand-sparkles": (
        '<path d="m21.64 3.64-1.28-1.28a1.21 1.21 0 0 0-1.72 0L2.36 18.64a1.21 1.21 0 0 0 0 '
        '1.72l1.28 1.28a1.2 1.2 0 0 0 1.72 0L21.64 5.36a1.2 1.2 0 0 0 0-1.72"/> <path d="m14 7 3 '
        '3"/> <path d="M5 6v4"/> <path d="M19 14v4"/> <path d="M10 2v2"/> <path d="M7 8H3"/> '
        '<path d="M21 16h-4"/> <path d="M11 3H9"/>'
    ),
    "x": '<path d="M18 6 6 18"/> <path d="m6 6 12 12"/>',

    "building-2": (
        '<path d="M6 22V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v18Z"/> <path d="M6 12H4a2 2 0 0 0-2 2v6a2'
        ' 2 0 0 0 2 2h2"/> <path d="M18 9h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-2"/> <path d="M10 '
        '6h4"/> <path d="M10 10h4"/> <path d="M10 14h4"/> <path d="M10 18h4"/>'
    ),
    "trees": (
        '<path d="M10 10v.2A3 3 0 0 1 8.9 16H5a3 3 0 0 1-1-5.8V10a3 3 0 0 1 6 0Z"/> <path d="M7 '
        '16v6"/> <path d="M13 19v3"/> <path d="M12 19h8.3a1 1 0 0 0 .7-1.7L18 14h.3a1 1 0 0 0 '
        '.7-1.7L16 9h.2a1 1 0 0 0 .8-1.7L13 3l-1.4 1.5"/>'
    ),
    "droplets": (
        '<path d="M7 16.3c2.2 0 4-1.83 4-4.05 0-1.16-.57-2.26-1.71-3.19S7.29 6.75 7 5.3c-.29 '
        '1.45-1.14 2.84-2.29 3.76S3 11.1 3 12.25c0 2.22 1.8 4.05 4 4.05z"/> <path d="M12.56 '
        '6.6A10.97 10.97 0 0 0 14 3.02c.5 2.5 2 4.9 4 6.5s3 3.5 3 5.5a6.98 6.98 0 0 1-11.91 '
        '4.97"/>'
    ),
    "car": (
        '<path d="M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 '
        '10s-1.3-1.4-2.2-2.3c-.5-.4-1.1-.7-1.8-.7H5c-.6 0-1.1.4-1.4.9l-1.4 2.9A3.7 3.7 0 0 0 2 '
        '12v4c0 .6.4 1 1 1h2"/> <circle cx="7" cy="17" r="2"/> <path d="M9 17h6"/> <circle '
        'cx="17" cy="17" r="2"/>'
    ),
    "wheat": (
        '<path d="M2 22 16 8"/> <path d="M3.47 12.53 5 11l1.53 1.53a3.5 3.5 0 0 1 0 4.94L5 '
        '19l-1.53-1.53a3.5 3.5 0 0 1 0-4.94Z"/> <path d="M7.47 8.53 9 7l1.53 1.53a3.5 3.5 0 0 1 0'
        ' 4.94L9 15l-1.53-1.53a3.5 3.5 0 0 1 0-4.94Z"/> <path d="M11.47 4.53 13 3l1.53 1.53a3.5 '
        '3.5 0 0 1 0 4.94L13 11l-1.53-1.53a3.5 3.5 0 0 1 0-4.94Z"/> <path d="M20 2h2v2a4 4 0 0 '
        '1-4 4h-2V6a4 4 0 0 1 4-4Z"/> <path d="M11.47 17.47 13 19l-1.53 1.53a3.5 3.5 0 0 1-4.94 '
        '0L5 19l1.53-1.53a3.5 3.5 0 0 1 4.94 0Z"/> <path d="M15.47 13.47 17 15l-1.53 1.53a3.5 3.5'
        ' 0 0 1-4.94 0L9 15l1.53-1.53a3.5 3.5 0 0 1 4.94 0Z"/> <path d="M19.47 9.47 21 11l-1.53 '
        '1.53a3.5 3.5 0 0 1-4.94 0L13 11l1.53-1.53a3.5 3.5 0 0 1 4.94 0Z"/>'
    ),
    "zap": (
        '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 '
        '10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>'
    ),
    "trophy": (
        '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/> <path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/> '
        '<path d="M4 22h16"/> <path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 '
        '22"/> <path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/> <path '
        'd="M18 2H6v7a6 6 0 0 0 12 0V2Z"/>'
    ),
    "plane": (
        '<path d="M17.8 19.2 16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 '
        '6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 '
        '5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z"/>'
    ),
    "factory": (
        '<path d="M2 20a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8l-7 5V8l-7 5V4a2 2 0 0 0-2-2H4a2 2 0 0 '
        '0-2 2Z"/> <path d="M17 18h1"/> <path d="M12 18h1"/> <path d="M7 18h1"/>'
    ),
    "star": (
        '<path d="M11.525 2.295a.53.53 0 0 1 .95 0l2.31 4.679a2.123 2.123 0 0 0 1.595 '
        '1.16l5.166.756a.53.53 0 0 1 .294.904l-3.736 3.638a2.123 2.123 0 0 0-.611 1.878l.882 '
        '5.14a.53.53 0 0 1-.771.56l-4.618-2.428a2.122 2.122 0 0 0-1.973 0L6.396 21.01a.53.53 0 0 '
        '1-.77-.56l.881-5.139a2.122 2.122 0 0 0-.611-1.879L2.16 9.795a.53.53 0 0 1 '
        '.294-.906l5.165-.755a2.122 2.122 0 0 0 1.597-1.16z"/>'
    ),
    "layout-grid": (
        '<rect width="7" height="7" x="3" y="3" rx="1"/> <rect width="7" height="7" x="14" y="3" '
        'rx="1"/> <rect width="7" height="7" x="14" y="14" rx="1"/> <rect width="7" height="7" '
        'x="3" y="14" rx="1"/>'
    ),
    "folder": (
        '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 '
        '7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>'
    ),
    "flame": (
        '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5'
        ' 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 '
        '2.5z"/>'
    ),
}


FALLBACK = {
    "search": "search", "globe": "globe", "map-pin": "pin", "layers": "layers",
    "palette": "pencil", "tag": "pencil", "locate-fixed": "expand", "camera": "camera",
    "file-down": "download", "layout-template": "layout", "cog": "gear",
    "square-terminal": "terminal", "braces": "code", "database": "table", "table-2": "table",
    "chart-column": "chart", "file-text": "file", "message-circle-question": "chat_bubble",
    "trash-2": "trash", "pencil": "pencil", "crosshair": "crs", "ruler": "measure",
    "route": "route", "puzzle": "puzzle", "mountain": "terrain", "grid-3x3": "hexgrid",
    "wand-sparkles": "sparkles", "scan": "expand", "check": "check", "list-checks": "check", "x": "close",
    "minus": "dash", "triangle-alert": "warning", "calculator": "table",
    "square-dashed-mouse-pointer": "polygon", "link": "link", "book-open": "book",
    "filter": "funnel", "combine": "merge", "folder-open": "folder", "eye": "eye",
    "image": "image", "sparkle": "spark", "arrow-up": "arrow_up", "corner-down-left": "undo",
    "refresh-cw": "undo", "undo-2": "undo", "redo-2": "redo",
    "building-2": "home", "trees": "terrain", "droplets": "layers", "car": "route",
    "wheat": "terrain", "zap": "bolt", "trophy": "star", "plane": "route",
    "factory": "home", "star": "star", "layout-grid": "layout", "folder": "layers",
    "flame": "spark", "history": "undo",
}

_RENDERER = None


def _renderer_class():

    global _RENDERER
    if _RENDERER is None:
        try:
            from qgis.PyQt.QtSvg import QSvgRenderer
            _RENDERER = QSvgRenderer
        except ImportError:
            _RENDERER = False
    return _RENDERER


def is_lucide(name: str) -> bool:
    return str(name or "").startswith(PREFIX)


def fallback(name: str) -> str:

    return FALLBACK.get(str(name or "")[len(PREFIX):], "circle")


def svg(name: str, colour: QColor, size: float) -> bytes:

    key = str(name or "")[len(PREFIX):]
    hex_colour = colour.name()


    stroke = 2.4 if size <= 13 else 2.2 if size <= 16 else 2.0
    body = _SPARKLE.format(colour=hex_colour) if key == "sparkle" else SHAPES.get(key, "")
    opacity = colour.alphaF()
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{hex_colour}" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round" opacity="{opacity:.3f}">{body}</svg>'
    ).encode()


def render(painter: QPainter, name: str, colour: QColor, size: float) -> bool:


    renderer_class = _renderer_class()
    key = str(name or "")[len(PREFIX):]
    if not renderer_class or (key != "sparkle" and key not in SHAPES):
        return False
    renderer = renderer_class(QByteArray(svg(name, colour, size)))
    if not renderer.isValid():
        return False
    renderer.render(painter, QRectF(0, 0, size, size))
    return True


__all__ = ["PREFIX", "SHAPES", "fallback", "is_lucide", "render", "svg"]
