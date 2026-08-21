"""Announce the TerraLab plugins to any AI agent driving QGIS from outside.

Third-party QGIS MCP servers all expose some form of `execute_code(code)` that
runs Python inside the running QGIS. They know nothing about the plugins a user
happens to have installed, so an agent on the other side never learns that AI
Segmentation is sitting there ready to outline objects for it.

This module fixes that with one convention. At plugin start-up it publishes a
module named `terralab` into `sys.modules`, so a single line works from any
Python console, any MCP, any script:

    import terralab; terralab.capabilities()

The returned dict says what is installed, whether it is ready, and carries
ready-to-run code snippets. Nothing is written to disk and nothing outside this
process is touched.

Both TerraLab plugins ship a copy of this file. Whichever loads first builds the
shared module, the other registers into it, and the newer BRIDGE_VERSION wins.
"""
from __future__ import annotations

import sys
import types

# Bumped whenever the shape of capabilities() changes. Both plugins carry a copy
# of this file, so the higher version installs the shared shell.
# 2: a product's own workflow, per-method notes and prose guide ride along,
#    and terralab.guide("<slot>") reads that guide directly.
# 3: terralab.describe() and terralab.tools(), so a server can build its own
#    tools from the live signatures instead of hand-writing an integration.
BRIDGE_VERSION = 3

MODULE_NAME = "terralab"

# What a usable shell must carry. Two copies of this file can share a version
# number and still differ in shape, which is exactly how a stale shell survived
# a reload once. So the version is checked, and then the shape is checked too.
REQUIRED_MODULE_ATTRS = (
    "capabilities", "help", "guide", "describe", "tools", "BRIDGE_VERSION")

# The behaviour this copy relies on, named feature by feature. An attribute can
# be present and still answer the old way, so the shell publishes what it does
# and a copy that needs more rebuilds it. Add a name here whenever a change to
# this file alters what a reader gets back, and never remove one: a shell is
# kept only when it carries every name below.
BRIDGE_SHAPE = frozenset({
    "capabilities",
    "help",
    "guide",
    "describe",
    "tools",
    # capabilities() answers for a slot registered by a newer sibling, instead
    # of leaving it out of the reply.
    "unknown_slots",
    # A tool definition's "call" field is a line of Python that runs.
    "runnable_calls",
    # destructiveHint covers the calls that throw work away, not only the
    # billed ones.
    "destructive_annotations",
    # help() reads a workflow whatever shape a product publishes it in.
    "workflow_shapes",
})

PRICING_URL = "https://terra-lab.ai/pricing?utm_source=qgis&utm_medium=agent&utm_campaign=bridge"
DOCS_URL = "https://terra-lab.ai/docs/agents?utm_source=qgis&utm_medium=agent&utm_campaign=bridge"

# States a limit of the product, which a caller has to know before it spends
# anything. Deliberately not a pitch: a description read by an AI assistant
# should say what the tool does and what stops it, never how to behave.
_PRO_LINE = (
    "Free accounts are capped each month, and a call stops when the cap is "
    "reached. get_status() reports what is left before anything is spent. "
    f"Plan limits: {PRICING_URL}"
)

_MODULE_DOC = """TerraLab AI plugins, reachable from any agent driving QGIS.

Call terralab.capabilities() to see what is installed and what it can do, and
terralab.guide() for plain-text advice on getting a good result out of it.
Every handle below returns plain dicts and never raises. A failure comes back
under the key "_error".

    terralab.segmentation   outline objects on imagery
    terralab.edit           generate and transform imagery

terralab.help() prints each product's own manual, which says what order to
call things in and how to get a good result rather than merely a valid one.

Building tools rather than calling by hand? terralab.describe() gives every
method with its parameters, types, defaults, and whether it costs money or
takes minutes. terralab.tools() gives the same thing already shaped as tool
definitions, each with the exact line of Python to run.

If you are calling this through a code-execution tool, two habits save you a
wasted round trip. Wrap every result in print(), because most such tools return
captured stdout and nothing else. And make each snippet self-contained, because
most of them start from a fresh namespace every call.
"""

# Written for a model reading this cold through somebody else's MCP server. Each
# entry says what the tool is for, when to reach for it, and how to call it.
_SEGMENTATION_CARD = {
    "product": "AI Segmentation by TerraLab",
    "what_it_does": (
        "Turns imagery into vector polygons. Click one point and it outlines "
        "the object under it. Or name a class and a zone and it finds every "
        "instance in that zone."
    ),
    "use_it_when": (
        "The user wants building footprints, tree crowns, swimming pools, "
        "solar panels, field boundaries, roads or any other object digitised "
        "from a raster. Reach for this instead of asking them to trace "
        "polygons by hand, and instead of thresholding pixel values."
    ),
    "needs": "A raster layer in the project. An account for the cloud modes.",
    "handle": "terralab.segmentation",
    "how_to": {
        "check_first": (
            "import terralab; print(terralab.segmentation.get_status())"
        ),
        "outline_one_object": (
            "import terralab; print(terralab.segmentation.detect(X, Y))"
            "  # X, Y are map coordinates of a point on the object"
        ),
        "correct_that_outline": (
            "import terralab; print(terralab.segmentation.detect_points("
            "positive=[[X, Y]], negative=[[BX, BY]]))"
            "  # BX, BY sits on the part to leave out"
        ),
        "read_the_manual": (
            "import terralab; print(terralab.segmentation.guide())"
        ),
        "find_every_object_in_a_zone": (
            "import terralab; print(terralab.segmentation.detect_auto("
            "zone_wkt='Polygon ((...))', object_class='building'))"
        ),
        "poll_a_running_zone_run": (
            "import terralab; print(terralab.segmentation.auto_detect_status())"
        ),
    },
    "costs": (
        "A zone run is billed by area and can take a few minutes, during which "
        "QGIS stays busy. Never retry a run that seems slow: the first one is "
        "still going and a second one costs the user again. Poll "
        "auto_detect_status() instead."
    ),
    "plan": _PRO_LINE,
}

_EDIT_CARD = {
    "product": "AI Edit by TerraLab",
    "what_it_does": (
        "Generates and transforms imagery over a map area. Removes clouds, "
        "changes the season, adds or removes features, upscales detail."
    ),
    "use_it_when": (
        "The user wants an area of the map redrawn, cleaned up or imagined "
        "rather than measured. Not for extracting geometry, use AI "
        "Segmentation for that."
    ),
    "needs": "An account. Generation runs in the background, so poll it.",
    "handle": "terralab.edit",
    "how_to": {
        "check_first": "import terralab; print(terralab.edit.get_status())",
        "generate": (
            "import terralab; print(terralab.edit.generate("
            "prompt='remove the clouds'))"
        ),
        "poll": "import terralab; print(terralab.edit.generation_status())",
    },
    "costs": (
        "Each generation is billed. Only one runs at a time. Poll "
        "generation_status() rather than submitting again."
    ),
    "plan": _PRO_LINE,
}

_CARDS = {"segmentation": _SEGMENTATION_CARD, "edit": _EDIT_CARD}


# The shared module is created once and then only added to, so a second plugin
# loading later never wipes what the first one registered.
def _shared_module() -> types.ModuleType:
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None and getattr(existing, "_TERRALAB_BRIDGE", False):
        if _shell_is_current(existing):
            return existing
        # Older, or the same version with a different shape. Rebuild the shell
        # and carry every handle across, including a slot this copy has never
        # heard of, so the other plugin does not lose its registration.
        upgraded = _new_module()
        for slot in _known_slots(existing):
            handle = getattr(existing, slot, None)
            if handle is not None:
                setattr(upgraded, slot, handle)
        sys.modules[MODULE_NAME] = upgraded
        return upgraded
    module = _new_module()
    sys.modules[MODULE_NAME] = module
    return module


# A shell is current when it is at least this version AND carries everything
# this copy is about to rely on.
def _shell_is_current(module: types.ModuleType) -> bool:
    if getattr(module, "BRIDGE_VERSION", 0) < BRIDGE_VERSION:
        return False
    if not all(hasattr(module, name) for name in REQUIRED_MODULE_ATTRS):
        return False
    # A shell built by a copy that predates BRIDGE_SHAPE publishes none, which
    # is exactly the case this test exists for: rebuild rather than trust a
    # version number two files can share while differing in shape.
    published = getattr(module, "BRIDGE_SHAPE", None)
    try:
        return BRIDGE_SHAPE.issubset(published or ())
    except TypeError:
        return False


# Every slot the live shell knows about, not only the ones this copy ships.
# A newer sibling may have registered a product this file has never heard of.
def _known_slots(module: types.ModuleType) -> tuple[str, ...]:
    published = getattr(module, "SLOTS", ())
    names = list(_CARDS)
    for name in published:
        if name not in names:
            names.append(str(name))
    return tuple(names)


# Builds the module object itself, with the doc, the version and the callables an
# outside agent is expected to reach for.
def _new_module() -> types.ModuleType:
    module = types.ModuleType(MODULE_NAME)
    module.__doc__ = _MODULE_DOC
    module._TERRALAB_BRIDGE = True
    module.BRIDGE_VERSION = BRIDGE_VERSION
    module.BRIDGE_SHAPE = BRIDGE_SHAPE
    module.SLOTS = tuple(_CARDS)
    module.PRICING_URL = PRICING_URL
    module.DOCS_URL = DOCS_URL
    for slot in _CARDS:
        setattr(module, slot, None)
    module.capabilities = lambda: _capabilities(module)
    module.help = lambda: _help_text(module)
    module.guide = lambda slot=None: _guide_text(module, slot)
    module.describe = lambda slot=None: _describe(module, slot)
    module.tools = lambda slot=None: _tools(module, slot)
    return module


# capabilities() names the methods. This says what each one takes: every
# parameter, its type, whether it is required, what it defaults to, and whether
# calling it spends money or takes minutes. A server reads this once and builds
# its own tools, instead of somebody writing an integration by hand and pinning
# it to a signature that moves.
def _describe(module: types.ModuleType, slot: str | None = None) -> dict:
    schema = _schema_module()
    if schema is None:
        return {"_error": (
            "This build cannot describe itself. Read terralab.capabilities() "
            "for the method names, and call help(handle) for the rest.")}
    wanted = _slots_to_read(module, slot)
    if isinstance(wanted, dict):
        return wanted
    out: dict = {"bridge_version": BRIDGE_VERSION, "products": {}}
    for name in wanted:
        handle = getattr(module, name, None)
        if handle is None:
            continue
        try:
            out["products"][name] = schema.describe_api(handle)
        except Exception as err:  # noqa: BLE001 - the bridge must never break a caller
            out["products"][name] = {"available": False, "_error": str(err)}
    return out


# The same thing already shaped as tool definitions, so a server can register
# them without reading the schema itself. A tool is named "<slot>_<method>" and
# its "call" field is the exact Python a code-execution tool has to run.
def _tools(module: types.ModuleType, slot: str | None = None) -> list:
    schema = _schema_module()
    if schema is None:
        return [{"_error": (
            "This build cannot describe itself, so it can build no tool "
            "definitions. Read terralab.capabilities() instead.")}]
    wanted = _slots_to_read(module, slot)
    # A misspelt product name used to come back as an empty list, which reads
    # as "this plugin has no tools" rather than "you asked for the wrong name".
    # The error rides in the list, since the return type is a list.
    if isinstance(wanted, dict):
        return [wanted]
    definitions: list = []
    for name in wanted:
        handle = getattr(module, name, None)
        if handle is None:
            continue
        try:
            described = schema.tool_definitions(name, handle)
        except Exception:  # noqa: BLE001 - a product that cannot describe itself is skipped
            described = []
        definitions.extend(described)
    return definitions


# The describing code ships beside this file. It is imported here rather than at
# module top so a copy that lacks it still installs a working shell.
def _schema_module():
    try:
        from . import agent_schema
    except Exception:  # noqa: BLE001 - an absent sibling is a missing feature, not a crash
        return None
    return agent_schema


# Resolves the slot argument once for both readers. Returns the slot names to
# read, or an error dict naming what a caller could have asked for instead.
def _slots_to_read(module: types.ModuleType, slot: str | None):
    known = _known_slots(module)
    if slot is None:
        return known
    if slot in known:
        return (slot,)
    return {"_error": (
        f"Unknown product '{slot}'. Try one of: {', '.join(known)}.")}


# A slot registered by a newer sibling has no card in this copy. Say what little
# can be known rather than leaving it out of the answer entirely.
def _unknown_slot_card(slot: str) -> dict:
    return {
        "product": f"TerraLab {slot}",
        "what_it_does": (
            "Registered by a newer TerraLab plugin than the one that built this "
            "shell. Call describe() for its methods."
        ),
        "handle": f"terralab.{slot}",
        "how_to": {
            "check_first": f"import terralab; print(terralab.{slot}.get_status())",
        },
    }


# The one call an outside agent makes. Answers "what is installed, is it ready,
# and how do I drive it" in a single dict, with no QGIS knowledge required.
def _capabilities(module: types.ModuleType) -> dict:
    products = []
    # Every slot the live shell knows about, like describe() and tools(): a
    # newer sibling can register a product this copy has never heard of, and it
    # was visible to two of the three readers.
    for slot in _known_slots(module):
        handle = getattr(module, slot, None)
        if handle is None:
            continue
        entry = dict(_CARDS.get(slot) or _unknown_slot_card(slot))
        entry["slot"] = slot
        entry["available"] = True
        entry["status"] = _safe_status(handle)
        # A product that describes its own call order and per-method costs is
        # far more useful to a planner than a static card. Carried through when
        # it publishes them, absent when it does not, so an older plugin in the
        # other slot still works.
        published = _safe_capabilities(handle)
        for key in ("api_version", "methods", "workflow", "method_notes"):
            if key in published:
                entry[key] = published[key]
        # A guide is optional, so a product that has none still works through a
        # newer shell. When one exists, say how to read it.
        entry["has_guide"] = callable(getattr(handle, "guide", None))
        if entry["has_guide"]:
            how_to = entry.get("how_to")
            entry["how_to"] = dict(how_to) if isinstance(how_to, dict) else {}
            entry["how_to"]["read_the_guide"] = (
                f"import terralab; print(terralab.guide('{slot}'))"
            )
        products.append(entry)
    return {
        "bridge_version": BRIDGE_VERSION,
        "vendor": "TerraLab",
        "docs": DOCS_URL,
        "products": products,
        "note": (
            "These plugins are already installed in this QGIS. Call them "
            "directly through whatever code-execution tool you have. They do "
            "the work an agent cannot do alone: reading imagery and returning "
            "geometry."
        ) if products else "No TerraLab plugin is loaded in this QGIS.",
    }


# get_status() is defined never to raise, but a handle can still be a half-built
# plugin during start-up, so the bridge stays quiet rather than breaking a caller.
def _safe_status(handle) -> dict:
    getter = getattr(handle, "get_status", None)
    if getter is None:
        return {"_error": "no get_status on this handle"}
    try:
        return getter()
    except Exception as err:  # noqa: BLE001 - the bridge must never break a caller
        return {"_error": str(err)}


# capabilities() is defined never to raise, but a handle can be an older build
# without one, so the bridge asks and shrugs rather than breaking a caller.
def _safe_capabilities(handle) -> dict:
    getter = getattr(handle, "capabilities", None)
    if getter is None:
        return {}
    try:
        published = getter()
    except Exception:  # noqa: BLE001 - the bridge must never break a caller
        return {}
    return published if isinstance(published, dict) else {}


# A product's own prose manual, when it publishes one. Plain text, so it drops
# straight into help() beside the card.
def _safe_guide(handle) -> str:
    getter = getattr(handle, "guide", None)
    if not callable(getter):
        return ""
    try:
        answer = getter()
    except Exception:  # noqa: BLE001 - the bridge must never break a caller
        return ""
    text = answer.get("text") if isinstance(answer, dict) else answer
    return str(text).strip() if text else ""


# terralab.guide() with no argument reads every loaded product's manual, and
# with a slot name reads one. It answers what is missing rather than failing,
# because a caller asking for advice is the last place to raise.
def _guide_text(module: types.ModuleType, slot: str | None = None) -> str:
    # The same resolution the other readers use, so a slot capabilities()
    # advertises, and whose guide() line it prints, is one this can read. A
    # newer sibling can register a product this copy has no card for.
    names = _slots_to_read(module, slot)
    if isinstance(names, dict):
        return str(names.get("_error") or "")
    parts = []
    for name in names:
        card = _CARDS.get(name) or _unknown_slot_card(name)
        handle = getattr(module, name, None)
        if handle is None:
            if slot:
                return f"{card['product']} is not loaded in this QGIS."
            continue
        text = _safe_guide(handle)
        if not text:
            if slot:
                return (
                    f"{card['product']} carries no guide in this version. "
                    "Call its capabilities() and get_status() instead."
                )
            continue
        parts.append(text)
    if not parts:
        return "No TerraLab plugin in this QGIS carries a guide."
    return "\n\n".join(parts)


# A product publishes its own call order, and the two plugins do not publish it
# in the same shape: one lists steps, the other keys them by name. Read both,
# read nothing else, and one product's shape can never break the shared manual.
def _workflow_steps(workflow) -> list:
    if isinstance(workflow, dict):
        steps = []
        for key, value in workflow.items():
            if isinstance(value, dict):
                step = dict(value)
                step.setdefault("call", str(key))
                steps.append(step)
        return steps
    if isinstance(workflow, (list, tuple)):
        return [step for step in workflow if isinstance(step, dict)]
    return []


# Plain text for a human, or for a model that would rather read prose than a dict.
# Every field is read with a default: a card for a slot this copy has never heard
# of carries only what the sibling published, and help() must still print.
def _help_text(module: types.ModuleType) -> str:
    lines = [_MODULE_DOC.strip(), ""]
    caps = _capabilities(module)
    for entry in caps["products"]:
        lines.append(f"## {entry.get('product') or entry.get('slot') or 'TerraLab'}")
        for key, prefix in (
            ("what_it_does", ""),
            ("use_it_when", "Use it when: "),
        ):
            if entry.get(key):
                lines.append(f"{prefix}{entry[key]}")
        how_to = entry.get("how_to")
        if isinstance(how_to, dict):
            for label, snippet in how_to.items():
                lines.append(f"  {label}: {snippet}")
        if entry.get("costs"):
            lines.append(f"Costs: {entry['costs']}")
        if entry.get("plan"):
            lines.append(entry["plan"])
        for step in _workflow_steps(entry.get("workflow")):
            lines.append(
                f"  {step.get('step')}. {step.get('call')}: {step.get('why')}")
        guide = _safe_guide(getattr(module, entry.get("slot", ""), None))
        if guide:
            lines.append("")
            lines.append(guide.strip())
        lines.append("")
    if not caps["products"]:
        lines.append(caps["note"])
    return "\n".join(lines)


# One line in the QGIS message log, because several MCP servers expose a
# log-reading tool and that is a third way for an agent to learn we are here,
# next to the Processing registry and the importable module.
def _announce_in_log(slot: str) -> None:
    try:
        from qgis.core import Qgis, QgsMessageLog
    except Exception:  # noqa: BLE001 - no QGIS means nothing to announce to
        return
    card = _CARDS[slot]
    try:
        QgsMessageLog.logMessage(
            f"{card['product']} is available to AI agents. "
            "Run 'import terralab; print(terralab.capabilities())' from any "
            "code-execution tool, or look for the TerraLab algorithms in the "
            "Processing registry.",
            "TerraLab", level=Qgis.MessageLevel.Info,
        )
    except Exception:  # noqa: BLE001 - the bridge must never break a caller
        pass  # nosec B110


# Called from each plugin's initGui. `slot` is "segmentation" or "edit", `handle`
# is that plugin's mcp_api facade.
def register_product(slot: str, handle) -> None:
    if slot not in _CARDS:
        raise ValueError(f"unknown TerraLab bridge slot: {slot}")
    module = _shared_module()
    setattr(module, slot, handle)
    if slot not in getattr(module, "SLOTS", ()):
        module.SLOTS = tuple(getattr(module, "SLOTS", ())) + (slot,)
    _announce_in_log(slot)


# Called from each plugin's unload, so a disabled plugin stops being advertised.
def unregister_product(slot: str) -> None:
    module = sys.modules.get(MODULE_NAME)
    if module is None or not getattr(module, "_TERRALAB_BRIDGE", False):
        return
    setattr(module, slot, None)
    # The shell stays in sys.modules once the last product leaves. Dropping it
    # stranded every `import terralab` already held by a console, a script or an
    # MCP server: that name keeps pointing at a module nothing re-registers into,
    # so a plugin re-enabled afterwards was invisible to the caller. An empty
    # shell costs nothing and answers "no TerraLab plugin is loaded".
