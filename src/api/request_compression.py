"""How a request body travels: plain bytes, or gzipped when the server asks.

Two bodies the plugin sends are large, and both are mostly base64: the crop
handed over before a click, and every tile of an Automatic run. Base64 spends
four bytes on every three, and deflate hands that third back, so the same
picture reaches the server in fewer bytes and the upload that the user waits
through is shorter.

The encoding of a body is a wire contract, and a server that cannot inflate
one reads a gzip stream as JSON and refuses the whole request, which on these
routes costs a credit and a click. The dial can turn the whole thing off for
the fleet, a first refusal pins one session back to plain bytes, and every
failure to compress falls through to the plain body.

Pure Python and no Qt, so both transports can import it: the blocking client
and the wait that keeps the map painting.
"""
from __future__ import annotations

import gzip

# Below this a body is round trips, not bytes: the header, the handshake and
# the wait dwarf whatever deflate could save, and the CPU is spent for nothing.
# Above it sit the geometry bodies as well as the pictures, and a slow uplink
# feels every kilobyte of those.
_COMPRESS_FLOOR_BYTES = 8 * 1024

# Level 1. What fills a large body is base64 over bytes that are already
# compressed pictures, so the deflate window has little left to find and the
# higher levels cost several times the time for a fraction of a percent.
_COMPRESS_LEVEL = 1

# A body the server could not read at all comes back 400, with nothing behind
# it: a route that never parsed the request has no field to name and no JSON
# to answer with, and a gateway that predates the encoding sends its own page
# or nothing at all. An ordinary validation failure comes back 422, and a
# route that did read the body and disagreed with it answers 400 with its own
# JSON. That last one is an answer, so the status alone is not the test: the
# shape of the answer is the other half of it.
_BODY_REFUSED_STATUS = 400

# Set the first time a server refuses a body it could not read, and never
# unset: the server behind this session is what it is, and asking it again in
# the same form would cost every later request its first answer. Module level
# rather than on one object, because the click, the crop hand-over and the tile
# submit each travel on their own object and one refusal has to reach all three.
_gzip_refused = False


def gzip_request_refused() -> bool:
    """Whether a server has already refused a compressed body this session."""
    return _gzip_refused


def note_gzip_request_refused() -> None:
    """Pin the session to plain bodies. Safe to call more than once."""
    global _gzip_refused
    _gzip_refused = True


def gzip_requests_allowed() -> bool:
    """Whether the served dial lets this client compress.

    Wrapped so a broken dial read costs the compression, never the request:
    anything that goes wrong here sends the body plain, which every server
    reads.
    """
    if _gzip_refused:
        return False
    try:
        from ..core.server_dials import gzip_request_bodies_enabled

        return gzip_request_bodies_enabled()
    except Exception:  # noqa: BLE001 -- a dial must never break a request  # nosec B110
        return False


def packed_request_body(body: bytes) -> tuple[bytes, bool]:
    """One request body ready to send: ``(bytes, whether they are gzipped)``.

    The plain bytes come back untouched whenever the dial is off, a server has
    already refused the form, the body is under the floor, deflate did not
    actually make it smaller, or anything at all went wrong. A caller that gets
    True sets ``Content-Encoding: gzip`` on the request and nothing else: Qt
    takes the length from the byte array it is handed.
    """
    if not body or len(body) < _COMPRESS_FLOOR_BYTES:
        return body, False
    if not gzip_requests_allowed():
        return body, False
    try:
        packed = gzip.compress(body, _COMPRESS_LEVEL)
    except Exception:  # noqa: BLE001 -- an unpacked body is always sendable
        return body, False
    if len(packed) >= len(body):
        return body, False
    return packed, True


def answer_refused_the_body(answer, http_status: int | None = None,
                            body_was_json: bool | None = None) -> bool:
    """Whether this answer says the server could not read the body it was sent.

    Read only about a request that went out compressed, and only to decide
    whether to send it again plain. The inference service answers a body it
    could not open with a 400, and a body it opened but disagreed with
    (a missing field, a bad value) with a 422, so the status alone separates
    the two. A 400 never follows a charge on that service, which is what makes
    the plain re-send safe. The 400 carries a JSON body of its own, so the
    shape of the answer says nothing here; ``body_was_json`` is kept for the
    callers that already pass it.
    """
    del answer, body_was_json  # See _BODY_REFUSED_STATUS: the status decides.
    return http_status == _BODY_REFUSED_STATUS
