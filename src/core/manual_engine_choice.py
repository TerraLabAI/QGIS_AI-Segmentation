"""The settings key the old cloud data notice wrote, kept out of the way.

DORMANT (2026-08-15): nothing reads or writes it any more. The modal that
asked the question is gone, replaced by a permanent disclosure line in the
Semi-Auto engine card (see ui/dock/manual_engine.py).

The key is left where it lies in the profiles that carry it. Removing it would
buy nothing and cost a migration. Frozen literal for the same reason: a rename
would make an old profile's answer unreadable to anything that ever looks
again.

An earlier build also stored whether the user had been SHOWN a chooser screen,
under ``AISegmentation/manual_engine_chosen``. That screen went the same way,
and so did its key.
"""
from __future__ import annotations

MANUAL_CLOUD_CONSENT_KEY = "AISegmentation/manual_cloud_consent"
