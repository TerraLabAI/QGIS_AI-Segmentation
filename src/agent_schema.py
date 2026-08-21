"""Describe the TerraLab APIs in a shape a machine can build tools from.

A human reads a panel. An agent reads a schema. This module turns whatever
facade a plugin exposes into a JSON-serialisable description: every method,
every parameter with its type and default, what the method is for, and the two
facts an agent has to know before it calls anything, whether the call costs the
user money and whether it is slow enough to need patience.

The point is that a third-party MCP server can read this once and generate its
own tool definitions from it, without anyone writing an integration by hand.
Everything is derived at runtime from the live object, so a plugin that gains a
method gains it here too, with no second list to keep in step.
"""
from __future__ import annotations

import inspect
from typing import Any

# Names whose calls spend the user's credits. An agent has to be able to tell
# these apart before it acts, not after the invoice.
BILLED_METHODS = {
    "detect",
    "detect_auto",
    "detect_points",
    "run_from_recipe",
    "generate",
}

# Names that can take minutes and hold QGIS while they run. An agent that knows
# this waits and warns instead of retrying, and a retry here is a second charge.
SLOW_METHODS = {
    "detect_auto",
    "run_from_recipe",
    "generate",
    "load_model",
}

# Names that only read. Safe to call at any time, in any order, for free. One
# set covers both plugins, since a name means the same thing in each and a
# method absent from a facade is simply never asked about.
READ_ONLY_METHODS = {
    # AI Segmentation
    "auto_detect_status",
    "describe_object_class",
    "export_recipe",
    "install_status",
    "list_object_classes",
    "refine_settings",
    "review_status",
    # AI Edit
    "generation_status",
    "get_account",
    "get_credits",
    "get_generation",
    "get_preset",
    "get_preset_families",
    "get_preset_family",
    "get_presets",
    "get_prompt_guidance",
    "get_resolutions",
    "get_session",
    "get_top_picks",
    "get_zone",
    "list_favorite_prompts",
    "list_generations",
    "list_recent_prompts",
    "list_references",
    "list_sessions",
    "list_versions",
    "markup_status",
    "search_presets",
    # Both
    "capabilities",
    "get_status",
    "guide",
}

# Names that throw work away. Deleting a session, dropping a reference, clearing
# a zone or stopping a run in flight all destroy something the user made, and
# none of them costs a credit, so "billed" does not catch a single one. An MCP
# client that auto-approves everything not marked destructive would delete a
# user's history without asking. One set covers both plugins, like the one
# above: a name absent from a facade is never asked about.
DESTRUCTIVE_METHODS = {
    # AI Segmentation
    "cancel_auto",
    "review_clear_corrections",
    "review_merge_objects",
    "review_remove_object",
    "review_undo_last",
    "undo_last_point",
    # AI Edit
    "cancel",
    "clear_references",
    "clear_zone",
    "delete_session",
    "remove_favorite_prompt",
    "remove_reference",
    "rename_session",
}

_PARAM_KINDS = {
    inspect.Parameter.POSITIONAL_ONLY: "positional",
    inspect.Parameter.POSITIONAL_OR_KEYWORD: "positional_or_keyword",
    inspect.Parameter.VAR_POSITIONAL: "varargs",
    inspect.Parameter.KEYWORD_ONLY: "keyword",
    inspect.Parameter.VAR_KEYWORD: "varkw",
}


# Turns a Python annotation into something that survives JSON, since annotations
# may be strings, classes or typing constructs depending on the module.
def _annotation_text(annotation: Any) -> str | None:
    if annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", None) or str(annotation)


# A default has to be JSON-safe. Anything exotic is reported by its repr rather
# than dropped, so a caller can still see that a default exists.
def _default_value(default: Any) -> Any:
    if default is inspect.Parameter.empty:
        return None
    if default is None or isinstance(default, (bool, int, float, str)):
        return default
    if isinstance(default, (list, tuple)):
        return [_default_value(item) for item in default]
    return repr(default)


# The first line of a docstring is the summary an agent shows or reasons over.
# The rest is kept separately so a caller can choose how much to read.
def _split_doc(func: Any) -> tuple[str, str]:
    doc = inspect.getdoc(func) or ""
    if not doc:
        return "", ""
    parts = doc.split("\n\n", 1)
    return parts[0].replace("\n", " ").strip(), (parts[1].strip() if len(parts) > 1 else "")


# Describes one method: what it takes, what it is for, what it costs.
def describe_method(name: str, func: Any) -> dict:
    entry: dict[str, Any] = {"name": name}
    summary, detail = _split_doc(func)
    if summary:
        entry["summary"] = summary
    if detail:
        entry["detail"] = detail
    params = []
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue
            params.append({
                "name": param_name,
                "kind": _PARAM_KINDS.get(param.kind, "unknown"),
                "type": _annotation_text(param.annotation),
                "required": param.default is inspect.Parameter.empty,
                "default": _default_value(param.default),
            })
    entry["parameters"] = params
    entry["billed"] = name in BILLED_METHODS
    entry["slow"] = name in SLOW_METHODS
    entry["read_only"] = name in READ_ONLY_METHODS
    entry["destructive"] = name in DESTRUCTIVE_METHODS or name in BILLED_METHODS
    return entry


# Describes a whole facade. `handle` is a plugin's mcp_api object. Methods are
# taken from its own PUBLIC_METHODS list when it publishes one, so a facade
# stays in charge of what it calls public.
def describe_api(handle: Any) -> dict:
    if handle is None:
        return {"available": False, "methods": []}
    names = _public_method_names(handle)
    methods = []
    for name in names:
        func = getattr(handle, name, None)
        if func is None or not callable(func):
            continue
        methods.append(describe_method(name, func))
    described: dict[str, Any] = {"available": True, "methods": methods}
    version = _api_version(handle)
    if version is not None:
        described["api_version"] = version
    return described


# A facade may publish its own list. Falling back to introspection means a build
# that predates the list is still described rather than reported as empty.
def _public_method_names(handle: Any) -> list[str]:
    declared = getattr(handle, "PUBLIC_METHODS", None)
    if isinstance(declared, (list, tuple)) and declared:
        return [str(name) for name in declared]
    module = inspect.getmodule(type(handle))
    declared = getattr(module, "PUBLIC_METHODS", None)
    if isinstance(declared, (list, tuple)) and declared:
        return [str(name) for name in declared]
    return sorted(
        name for name in dir(handle)
        if not name.startswith("_") and callable(getattr(handle, name, None))
    )


# The version travels with the description, so a caller that caches a schema can
# tell when it has gone stale.
def _api_version(handle: Any) -> int | None:
    getter = getattr(handle, "capabilities", None)
    if callable(getter):
        try:
            reported = getter()
        except Exception:  # noqa: BLE001 - describing must never break a caller
            reported = None
        if isinstance(reported, dict) and isinstance(reported.get("api_version"), int):
            return reported["api_version"]
    module = inspect.getmodule(type(handle))
    version = getattr(module, "API_VERSION", None)
    return version if isinstance(version, int) else None


# One tool definition per method, in the shape an MCP server registers. A
# third-party server can loop over this and expose our features as its own tools
# without anyone writing an adapter.
def tool_definitions(prefix: str, handle: Any) -> list[dict]:
    described = describe_api(handle)
    if not described.get("available"):
        return []
    tools = []
    for method in described["methods"]:
        properties = {}
        required = []
        for param in method["parameters"]:
            if param["kind"] in ("varargs", "varkw"):
                continue
            # A function cannot cross JSON. Advertising one asks a model to send
            # something it can never send, and the type test below reads the
            # word "bool" out of a callable's return and calls it a checkbox.
            if _is_callable_annotation(param["type"]):
                continue
            properties[param["name"]] = {
                "type": _json_type(param["type"]),
                "description": param["type"] or "",
            }
            if param["required"]:
                required.append(param["name"])
        tools.append({
            "name": f"{prefix}_{method['name']}",
            "description": method.get("summary", ""),
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
            "annotations": {
                "readOnlyHint": method["read_only"],
                "destructiveHint": method["destructive"],
            },
            "call": _call_line(prefix, method),
        })
    return tools


# The line a code-execution tool runs to reach this method. It carries the
# import and the print, because most such tools start from a fresh namespace and
# hand back captured stdout and nothing else. Required arguments are named in a
# trailing comment instead of being filled with a stand-in value: a stand-in is
# either unreadable to the method or, worse, readable, and then the printed line
# runs the method on a value nobody chose. Left as printed, the call is missing
# its arguments, and the facade answers by naming them.
def _call_line(prefix: str, method: dict) -> str:
    required = [
        param["name"] for param in method["parameters"]
        if param["required"] and param["kind"] not in ("varargs", "varkw")
    ]
    line = f"import terralab; print(terralab.{prefix}.{method['name']}())"
    if required:
        line += f"  # fill in: {', '.join(required)}"
    return line


# True for an annotation naming a function the caller has to pass in. Such a
# parameter belongs in the Python signature and never in a tool schema.
def _is_callable_annotation(annotation: str | None) -> bool:
    if not annotation:
        return False
    text = str(annotation).lower()
    return "callable" in text or "->" in text


# Maps a Python annotation onto the JSON Schema type an MCP client expects.
# Anything unrecognised becomes a string, which is always safe to send.
def _json_type(annotation: str | None) -> str:
    if not annotation:
        return "string"
    text = annotation.lower()
    # Containers are read first, and this order is the whole point. A
    # list[list[float]] holds the word "float", so a scalar test running first
    # calls it a number, and a server building a tool from that tells the model
    # to send one coordinate where a list of points belongs.
    if "list" in text or "tuple" in text or "sequence" in text:
        return "array"
    if "dict" in text or "mapping" in text:
        return "object"
    if "bool" in text:
        return "boolean"
    if "int" in text:
        return "integer"
    if "float" in text:
        return "number"
    return "string"
