



















from __future__ import annotations

import sys
import types







BRIDGE_VERSION = 3

MODULE_NAME = "terralab"




REQUIRED_MODULE_ATTRS = (
    "capabilities", "help", "guide", "describe", "tools", "BRIDGE_VERSION")






BRIDGE_SHAPE = frozenset({
    "capabilities",
    "help",
    "guide",
    "describe",
    "tools",


    "unknown_slots",

    "runnable_calls",


    "destructive_annotations",

    "workflow_shapes",


    "concise_describe",
})

PRICING_URL = "https://terra-lab.ai/pricing?utm_source=qgis&utm_medium=agent&utm_campaign=bridge"
DOCS_URL = "https://terra-lab.ai/docs/agents?utm_source=qgis&utm_medium=agent&utm_campaign=bridge"


def _pricing_url() -> str:

    from .core.server_dials import dial_url

    return dial_url("tuning.links.agent_pricing", PRICING_URL)


def _docs_url() -> str:

    from .core.server_dials import dial_url

    return dial_url("tuning.links.agent_docs", DOCS_URL)





_PRO_LINE_TEMPLATE = (
    "Free accounts are capped each month, and a call stops when the cap is "
    "reached. get_status() reports what is left before anything is spent. "
    "Plan limits: {url}"
)


def _pro_line() -> str:






    from .core.server_dials import dial_text

    served = dial_text("tuning.agent.bridge_card", "pro_line", 800)
    return served or _PRO_LINE_TEMPLATE.format(url=_pricing_url())


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
takes minutes; terralab.describe(detail="concise") is the same at about half
the length. terralab.tools() gives the same thing already shaped as tool
definitions, each with the exact line of Python to run.

If you are calling this through a code-execution tool, two habits save you a
wasted round trip. Wrap every result in print(), because most such tools return
captured stdout and nothing else. And make each snippet self-contained, because
most of them start from a fresh namespace every call.
"""


def _module_doc() -> str:





    from .core.server_dials import dial_text

    served = dial_text("tuning.agent", "module_doc", 4000)
    return served or _MODULE_DOC




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
            "zone_wkt='Polygon ((...))', object_class='building', wait=False))"
            "  # returns at once with a run_id"
        ),
        "poll_a_running_zone_run": (
            "import terralab; print(terralab.segmentation.auto_detect_status("
            "wait_s=45))  # returns as soon as the run ends, or after 45 s"
        ),
    },
    "costs": (
        "A zone run is billed by area and can take many minutes. Start it with "
        "wait=False and wait on it with auto_detect_status(wait_s=45), one "
        "call per 45 s. Never retry a run that seems slow: the first one is "
        "still going and a second one costs the user again."
    ),
    "plan": _PRO_LINE_TEMPLATE.format(url=PRICING_URL),
}


def _segmentation_card() -> dict:







    from .core.server_dials import dial_text

    card = dict(_SEGMENTATION_CARD)
    for field in ("what_it_does", "use_it_when", "needs", "costs"):
        served = dial_text("tuning.agent.bridge_card", field, 800)
        if served:
            card[field] = served
    card["plan"] = _pro_line()
    return card


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
    "plan": _PRO_LINE_TEMPLATE.format(url=PRICING_URL),
}

_CARDS = {"segmentation": _SEGMENTATION_CARD, "edit": _EDIT_CARD}




def _card_for(slot: str) -> dict | None:
    if slot == "segmentation":
        return _segmentation_card()
    return _CARDS.get(slot)




def _shared_module() -> types.ModuleType:
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None and getattr(existing, "_TERRALAB_BRIDGE", False):
        if _shell_is_current(existing):
            return existing



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




def _shell_is_current(module: types.ModuleType) -> bool:
    if getattr(module, "BRIDGE_VERSION", 0) < BRIDGE_VERSION:
        return False
    if not all(hasattr(module, name) for name in REQUIRED_MODULE_ATTRS):
        return False



    published = getattr(module, "BRIDGE_SHAPE", None)
    try:
        return BRIDGE_SHAPE.issubset(published or ())
    except TypeError:
        return False




def _known_slots(module: types.ModuleType) -> tuple[str, ...]:
    published = getattr(module, "SLOTS", ())
    names = list(_CARDS)
    for name in published:
        if name not in names:
            names.append(str(name))
    return tuple(names)




def _new_module() -> types.ModuleType:
    module = types.ModuleType(MODULE_NAME)
    module.__doc__ = _module_doc()
    module._TERRALAB_BRIDGE = True
    module.BRIDGE_VERSION = BRIDGE_VERSION
    module.BRIDGE_SHAPE = BRIDGE_SHAPE
    module.SLOTS = tuple(_CARDS)
    module.PRICING_URL = _pricing_url()
    module.DOCS_URL = _docs_url()
    for slot in _CARDS:
        setattr(module, slot, None)
    module.capabilities = lambda: _capabilities(module)
    module.help = lambda: _help_text(module)
    module.guide = lambda slot=None: _guide_text(module, slot)
    module.describe = lambda slot=None, detail="full": _describe(module, slot, detail)
    module.tools = lambda slot=None: _tools(module, slot)
    return module







def _describe(module: types.ModuleType, slot: str | None = None,
              detail: str = "full") -> dict:
    schema = _schema_module()
    if schema is None:
        return {"_error": (
            "This build cannot describe itself. Read terralab.capabilities() "
            "for the method names, and call help(handle) for the rest.")}
    levels = getattr(schema, "DESCRIBE_DETAIL_LEVELS", ("full",))
    if detail not in levels:
        return {"_error": (
            f"detail must be one of {', '.join(repr(v) for v in levels)}, "
            f"got {detail!r}.")}
    wanted = _slots_to_read(module, slot)
    if isinstance(wanted, dict):
        return wanted
    out: dict = {"bridge_version": BRIDGE_VERSION, "products": {}}
    for name in wanted:
        handle = getattr(module, name, None)
        if handle is None:
            continue
        try:
            out["products"][name] = (
                schema.describe_api(handle, detail=detail) if detail != "full"
                else schema.describe_api(handle))
        except Exception as err:  # noqa: BLE001
            out["products"][name] = {"available": False, "_error": str(err)}
    return out





def _tools(module: types.ModuleType, slot: str | None = None) -> list:
    schema = _schema_module()
    if schema is None:
        return [{"_error": (
            "This build cannot describe itself, so it can build no tool "
            "definitions. Read terralab.capabilities() instead.")}]
    wanted = _slots_to_read(module, slot)



    if isinstance(wanted, dict):
        return [wanted]
    definitions: list = []
    for name in wanted:
        handle = getattr(module, name, None)
        if handle is None:
            continue
        try:
            described = schema.tool_definitions(name, handle)
        except Exception:  # noqa: BLE001
            described = []
        definitions.extend(described)
    return definitions




def _schema_module():
    try:
        from . import agent_schema
    except Exception:  # noqa: BLE001
        return None
    return agent_schema




def _slots_to_read(module: types.ModuleType, slot: str | None):
    known = _known_slots(module)
    if slot is None:
        return known
    if slot in known:
        return (slot,)
    return {"_error": (
        f"Unknown product '{slot}'. Try one of: {', '.join(known)}.")}




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




def _capabilities(module: types.ModuleType) -> dict:
    products = []



    for slot in _known_slots(module):
        handle = getattr(module, slot, None)
        if handle is None:
            continue
        entry = dict(_card_for(slot) or _unknown_slot_card(slot))
        entry["slot"] = slot
        entry["available"] = True
        entry["status"] = _safe_status(handle)




        published = _safe_capabilities(handle)
        for key in ("api_version", "methods", "workflow", "method_notes"):
            if key in published:
                entry[key] = published[key]


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
        "docs": _docs_url(),
        "products": products,
        "note": (
            "These plugins are already installed in this QGIS. Call them "
            "directly through whatever code-execution tool you have. They do "
            "the work an agent cannot do alone: reading imagery and returning "
            "geometry."
        ) if products else "No TerraLab plugin is loaded in this QGIS.",
    }




def _safe_status(handle) -> dict:
    getter = getattr(handle, "get_status", None)
    if getter is None:
        return {"_error": "no get_status on this handle"}
    try:
        return getter()
    except Exception as err:  # noqa: BLE001
        return {"_error": str(err)}




def _safe_capabilities(handle) -> dict:
    getter = getattr(handle, "capabilities", None)
    if getter is None:
        return {}
    try:
        published = getter()
    except Exception:  # noqa: BLE001
        return {}
    return published if isinstance(published, dict) else {}




def _safe_guide(handle) -> str:
    getter = getattr(handle, "guide", None)
    if not callable(getter):
        return ""
    try:
        answer = getter()
    except Exception:  # noqa: BLE001
        return ""


    text = answer.get("text") if isinstance(answer, dict) else answer
    return str(text).strip() if text else ""





def _guide_text(module: types.ModuleType, slot: str | None = None) -> str:



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





def _help_text(module: types.ModuleType) -> str:
    lines = [_module_doc().strip(), ""]
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





def _announce_in_log(slot: str) -> None:
    try:
        from qgis.core import Qgis, QgsMessageLog
    except Exception:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001
        pass  # nosec B110




def register_product(slot: str, handle) -> None:
    if slot not in _CARDS:
        raise ValueError(f"unknown TerraLab bridge slot: {slot}")
    module = _shared_module()
    setattr(module, slot, handle)
    if slot not in getattr(module, "SLOTS", ()):
        module.SLOTS = tuple(getattr(module, "SLOTS", ())) + (slot,)
    _announce_in_log(slot)



def unregister_product(slot: str) -> None:
    module = sys.modules.get(MODULE_NAME)
    if module is None or not getattr(module, "_TERRALAB_BRIDGE", False):
        return
    setattr(module, slot, None)





