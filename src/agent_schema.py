












from __future__ import annotations

import inspect
from typing import Any



BILLED_METHODS = {
    "detect",
    "detect_auto",
    "detect_points",
    "run_from_recipe",
    "generate",
}



SLOW_METHODS = {
    "detect_auto",
    "run_from_recipe",
    "generate",
    "load_model",
}




READ_ONLY_METHODS = {

    "auto_detect_status",
    "describe_object_class",
    "export_recipe",
    "install_status",
    "list_object_classes",
    "refine_settings",
    "review_status",

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

    "capabilities",
    "get_status",
    "guide",
}







DESTRUCTIVE_METHODS = {

    "cancel_auto",
    "review_clear_corrections",
    "review_merge_objects",
    "review_remove_object",
    "review_undo_last",
    "undo_last_point",

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




def _annotation_text(annotation: Any) -> str | None:
    if annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", None) or str(annotation)




def _default_value(default: Any) -> Any:
    if default is inspect.Parameter.empty:
        return None
    if default is None or isinstance(default, (bool, int, float, str)):
        return default
    if isinstance(default, (list, tuple)):
        return [_default_value(item) for item in default]
    return repr(default)




def _split_doc(func: Any) -> tuple[str, str]:
    doc = inspect.getdoc(func) or ""
    if not doc:
        return "", ""
    parts = doc.split("\n\n", 1)
    return parts[0].replace("\n", " ").strip(), (parts[1].strip() if len(parts) > 1 else "")




def _is_section_header(lines: list[str], index: int) -> bool:
    line = lines[index]
    stripped = line.strip()
    if not stripped or line.startswith(" "):
        return False
    if index + 1 >= len(lines):
        return False
    underline = lines[index + 1].strip()
    return bool(underline) and set(underline) == {"-"}






def _param_descriptions(detail: str) -> dict[str, str]:
    if not detail:
        return {}
    lines = detail.splitlines()
    out: dict[str, list[str]] = {}
    in_params = False
    current: str | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _is_section_header(lines, i):
            in_params = stripped == "Parameters"
            current = None
            continue
        if not in_params or (stripped and set(stripped) == {"-"}):
            continue
        if not stripped:
            current = None
            continue
        if not line.startswith(" ") and ":" in stripped:
            current = stripped.split(":", 1)[0].strip()
            out[current] = []
        elif current is not None:
            out[current].append(stripped)
    return {name: " ".join(parts).strip() for name, parts in out.items() if parts}




def _leading_prose(detail: str) -> str:
    if not detail:
        return ""
    lines = detail.splitlines()
    prose = []
    for i, line in enumerate(lines):
        if _is_section_header(lines, i):
            break
        prose.append(line.strip())
    return " ".join(p for p in prose if p).strip()



def describe_method(name: str, func: Any) -> dict:
    entry: dict[str, Any] = {"name": name}
    summary, detail = _split_doc(func)
    if summary:
        entry["summary"] = summary
    if detail:
        entry["detail"] = detail
    param_text = _param_descriptions(detail)
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
                "description": param_text.get(param_name, ""),
                "required": param.default is inspect.Parameter.empty,
                "default": _default_value(param.default),
            })
    entry["parameters"] = params
    entry["billed"] = name in BILLED_METHODS
    entry["slow"] = name in SLOW_METHODS
    entry["read_only"] = name in READ_ONLY_METHODS
    entry["destructive"] = name in DESTRUCTIVE_METHODS or name in BILLED_METHODS
    return entry





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




def _api_version(handle: Any) -> int | None:
    getter = getattr(handle, "capabilities", None)
    if callable(getter):
        try:
            reported = getter()
        except Exception:  # noqa: BLE001
            reported = None
        if isinstance(reported, dict) and isinstance(reported.get("api_version"), int):
            return reported["api_version"]
    module = inspect.getmodule(type(handle))
    version = getattr(module, "API_VERSION", None)
    return version if isinstance(version, int) else None





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



            if _is_callable_annotation(param["type"]):
                continue




            properties[param["name"]] = {
                "type": _json_type(param["type"]),
                "description": param.get("description") or param["type"] or "",
            }
            if param["required"]:
                required.append(param["name"])
        summary = method.get("summary", "")
        extra = _leading_prose(method.get("detail", ""))
        description = f"{summary} {extra}".strip() if extra else summary
        tools.append({
            "name": f"{prefix}_{method['name']}",
            "description": description,
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









def _call_line(prefix: str, method: dict) -> str:
    required = [
        param["name"] for param in method["parameters"]
        if param["required"] and param["kind"] not in ("varargs", "varkw")
    ]
    line = f"import terralab; print(terralab.{prefix}.{method['name']}())"
    if required:
        line += f"  # fill in: {', '.join(required)}"
    return line




def _is_callable_annotation(annotation: str | None) -> bool:
    if not annotation:
        return False
    text = str(annotation).lower()
    return "callable" in text or "->" in text




def _json_type(annotation: str | None) -> str:
    if not annotation:
        return "string"
    text = annotation.lower()




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
