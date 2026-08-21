"""What an outside agent needs to know before it drives AI Segmentation.

`SegmentationMCPAPI.capabilities()` serves the machine-readable half of this
module (the call order and the per-method notes) and `guide()` serves the plain
text half. Both are written for a program that has never seen this plugin: they
say which call comes first, which calls spend money, which are slow, and how to
get a good result rather than merely a valid one.

Nothing here touches QGIS, so it can be read from any process.
"""
from __future__ import annotations


# The order a caller should work in. Every entry is one call plus the reason it
# comes where it does, so an agent can plan without reading any docstring.
def agent_workflow_steps() -> list[dict]:
    """The normal order of calls, as a list an agent can follow top to bottom.

    Each item has ``step`` (int), ``call`` (method name), ``why`` (one line)
    and ``optional`` (bool). A caller that follows this order never has to
    discover a precondition by getting an error back.
    """
    return [
        {
            "step": 1,
            "call": "get_status",
            "why": "Says whether the model is installed, loaded and pointed at "
                   "a raster. Every other call assumes this came back ready.",
            "optional": False,
        },
        {
            "step": 2,
            "call": "load_model",
            "why": "Only when get_status reports MODEL_NOT_LOADED. Takes a "
                   "while the first time in a session.",
            "optional": True,
        },
        {
            "step": 3,
            "call": "set_mode",
            "why": "'interactive' to click objects one at a time, 'automatic' "
                   "to sweep a zone for every instance of a class.",
            "optional": True,
        },
        {
            "step": 4,
            "call": "detect_points",
            "why": "Interactive route. Outline one object from a point, and "
                   "correct its shape with more points.",
            "optional": True,
        },
        {
            "step": 5,
            "call": "list_object_classes",
            "why": "Before any zone run. The validated words detect_auto "
                   "accepts as object_class, so pick one instead of guessing. "
                   "Free and instant.",
            "optional": True,
        },
        {
            "step": 6,
            "call": "detect_auto",
            "why": "Automatic route. Name a class and a zone, get every "
                   "instance. Pass confidence and refine here rather than "
                   "adjusting anything afterwards.",
            "optional": True,
        },
        {
            "step": 7,
            "call": "auto_detect_status",
            "why": "Poll a zone run instead of starting a second one. A run "
                   "that looks stuck is usually still working.",
            "optional": True,
        },
        {
            "step": 8,
            "call": "review_status",
            "why": "Only when a person left a finished run open in the panel. "
                   "A run started from this API saves itself and leaves no "
                   "review to adjust.",
            "optional": True,
        },
    ]


# Per-method facts a planner needs before it calls: what it spends, how long it
# blocks, and what has to exist in the project first.
def agent_method_notes() -> dict[str, dict]:
    """One entry per public method: cost, speed and preconditions.

    Keys are method names. Each value has ``spends`` (bool: the call can use up
    part of the account's monthly allowance), ``slow`` (bool: the call can
    block for minutes), ``needs_raster`` (bool) and ``summary`` (one line).
    """
    fast_free = {"spends": False, "slow": False, "needs_raster": False}
    return {
        "capabilities": dict(fast_free, summary="What this build can do."),
        "guide": dict(fast_free, summary="How to get good results, in prose."),
        "get_status": dict(fast_free, summary="Readiness and what to fix."),
        "install_status": dict(
            fast_free, summary="What is on disk. Read only, installs nothing."),
        "load_model": {
            "spends": False, "slow": True, "needs_raster": False,
            "summary": "Loads the on-device model. Returns a status, never hangs.",
        },
        "set_mode": dict(fast_free, summary="Switch interactive or automatic."),
        "list_object_classes": dict(
            fast_free,
            summary="The validated words detect_auto accepts as object_class."),
        "describe_object_class": dict(
            fast_free, summary="Everything known about one object class."),
        "detect": {
            "spends": True, "slow": False, "needs_raster": True,
            "summary": "One point, one object, saved straight to a GeoPackage.",
        },
        "detect_points": {
            "spends": True, "slow": False, "needs_raster": True,
            "summary": "Several points, one object. Negative points cut parts off.",
        },
        "detect_auto": {
            "spends": True, "slow": True, "needs_raster": True,
            "summary": "Sweeps a zone for every instance of a class. Minutes.",
        },
        "set_auto_zone": dict(
            fast_free, needs_raster=True, summary="Set the zone for the panel."),
        "auto_detect_status": dict(fast_free, summary="Poll a running zone run."),
        "cancel_auto": dict(
            fast_free, summary="Stop a run, keeping what it already produced."),
        "export_polygon": dict(
            fast_free, summary="Write one polygon to a GeoPackage layer."),
        "export_recipe": dict(fast_free, summary="Pack a run's intent into a token."),
        "run_from_recipe": {
            "spends": True, "slow": True, "needs_raster": True,
            "summary": "Replay a run from its token.",
        },
        "refine_settings": dict(
            fast_free, summary="Read the current shape-cleanup settings."),
        "apply_refine": dict(
            fast_free, summary="Reshape the objects of an open review."),
        "review_status": dict(fast_free, summary="What an open review holds."),
        "review_filter": dict(
            fast_free, summary="Re-filter an open review by confidence and size."),
        "set_display_mode": dict(fast_free, summary="Recolour an open review."),
        "review_remove_object": dict(
            fast_free, summary="Drop one object from an open review."),
        "review_merge_objects": dict(
            fast_free, summary="Join several objects into one. Free."),
        "review_undo_last": dict(fast_free, summary="Take back the last correction."),
        "review_clear_corrections": dict(
            fast_free, summary="Take back every correction of this review."),
        "undo_last_point": dict(
            fast_free, summary="Take back the last click of a panel session."),
    }


_GUIDE_TEXT = """AI Segmentation, for an agent driving it.

WHAT IT IS
Two ways to turn imagery into vector polygons.
Interactive: you give a point, it outlines the object under that point.
Automatic: you give a zone and a word, it finds every instance in the zone.
Interactive can run entirely on the user's computer. Automatic never can.

ORDER OF CALLS
1. get_status(). It tells you what is missing and what the user must do about
   it. Do not guess past it.
2. load_model() only if the status says the model is not loaded.
3. Then detect_points() for one object, or detect_auto() for a whole zone.
4. Poll a zone run with auto_detect_status(). Never start a second run because
   the first looks slow. The first is still going and the second costs again.

INTERACTIVE: POINTS
Put the first point near the middle of the object, not on its edge.
If the outline swallows a neighbour, add a negative point on the part you do
not want and call again. If it stops short, add a positive point on the part it
missed. Two or three points settle almost every shape. Ten do not fix what two
could not.
Each saved object counts against the account when the work runs in the cloud.
Work done on the user's own computer costs nothing.

AUTOMATIC: THE WORD YOU PASS
Do not guess the word. Call list_object_classes() first: it returns every
validated word this plugin knows, with the token to pass, the category it
belongs to, and a weak flag on the classes that name a kind of cover rather
than a countable object. describe_object_class("building") answers for one
word, and corrects a near miss ("buildings" is told to pass "building"). Both
are free. Pass the token field, never a translated label.
The word has to name something a person could point at. It works by contrast:
discrete objects that stand apart from their background. Car, truck, boat,
ship, aeroplane, train, building, house, storage tank, shipping container,
tree, swimming pool, tennis court all read well.
A word that names a machine rather than the shape on the ground reads badly.
So does a word for a kind of land rather than a thing on it.
Scene matters as much as the word. The same word can find nothing on a flat,
edge-to-edge surface and find a hundred objects where the same things sit apart
on contrasting ground. Before you decide a word is weak, try it once on a scene
where the objects are clearly separated.

AUTOMATIC: EXAMPLES INSTEAD OF A WORD
When no word fits, draw the answer instead. Pass exemplars: boxes around one or
two objects of the kind you want, in the raster layer's CRS, label 1 to find
more like it and label 0 to exclude. object_class may then be empty, but an
example-only run needs at least two positive boxes.
Examples are the right tool exactly where words are weak: shapes with no common
name, or a name that means the machine and not the mark it leaves.
An example run reads the whole zone as one image, so keep the zone modest or
the example becomes too small to recognise. It also costs far less than a
tiled run.

AUTOMATIC: FRAMING
Frame so the objects sit inside their surroundings with room to spare. Never
crop tight onto one object's texture: its edges then touch the frame and get
cut. A wider frame holds more whole objects and more contrast, and finds more.
When you are placing the zone from a name, geocode the exact feature, not the
town it is in. A town centre lands on the wrong thing and wastes the run.

AUTOMATIC: DETAIL AGAINST OBJECT SIZE
Detail is how many tiles the zone is cut into along its longer side.
Many small objects (trees, cars, animals) want HIGH detail: smaller tiles, so
each object is big enough in its tile to be seen.
One large object (a lake, a quarry, a bay) wants LOW detail, 1 or 2, so the
object is not sliced across tile edges.
Getting this backwards is the most common reason a run comes back thin.

AUTOMATIC: TOO MANY RESULTS
A run answers generously and lets you filter afterwards, for free. Pass a
higher confidence to detect_auto to keep less, or a lower one to keep more.
Words that name a single landmark (a bridge, a dam, a roundabout) tend to
return many fragments of the one thing. Keep the largest and drop the rest.
An object cut across two tiles comes back as two polygons; join them.

REFINING THE SHAPE
Refine settings clean up outlines after detection: simplify, keep a share of
the points, expand or contract, fill holes, square the corners, drop pieces
under a size. Pass them to detect_auto with refine= and they apply to that run.
Match the setting to the object. Square the corners of buildings; never square
a tree or a pond. Fill holes on a roof; keep them on a field with a pond in it.
Simplify a hand-sized outline gently, a field boundary harder.

CORRECTING WHAT CAME BACK
When a person leaves a finished run open in the panel, this API can adjust it:
change the confidence, recolour it, drop an object, join several into one, and
undo any of that. A run started from this API saves itself immediately, so
there is nothing left open to adjust. Decide before the run, not after.

WHAT THIS API WILL NOT DO
It never installs software and never signs the user in. If get_status says the
model is missing or the account is not active, tell the user what to click. It
is their machine and their account.
"""


# Plain prose for a model that would rather read a manual than a schema. Kept
# as one constant so capabilities() and guide() can never drift apart.
def agent_guide_text() -> str:
    """The prose manual: how to get good results, not merely valid ones.

    Returns one plain-text string, safe to print or to hand straight to a
    language model. It covers call order, how to choose the word passed to a
    zone run, when drawn examples beat a word, how framing changes a result,
    how detail should match the size of the objects, and what this API
    deliberately will not do.
    """
    return _GUIDE_TEXT
