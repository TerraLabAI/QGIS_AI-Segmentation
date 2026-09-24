












from __future__ import annotations

import inspect
import re
from typing import Any




DESCRIBE_DETAIL_LEVELS = ("full", "concise")



_TOOL_COST_CHARS = 300



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
    "review_objects",
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




def _section_text(detail: str, header: str) -> str:
    if not detail:
        return ""
    lines = detail.splitlines()
    body: list[str] = []
    inside = False
    for i, line in enumerate(lines):
        if _is_section_header(lines, i):
            if inside:
                break
            inside = line.strip() == header
            continue
        stripped = line.strip()
        if inside and stripped and set(stripped) != {"-"}:
            body.append(stripped)
    return " ".join(body).strip()







def _concise_detail(detail: str) -> str:
    prose = _leading_prose(detail)
    cost = _section_text(detail, "Cost")
    parts = [prose] if prose else []
    if cost:
        parts.append(f"Cost: {cost}")
    return " ".join(parts)




def _concise_param(param: dict) -> dict:
    out = {"name": param["name"], "type": param["type"],
           "required": param["required"]}
    if not param["required"]:
        out["default"] = param["default"]
    if param.get("description"):
        out["description"] = param["description"]
    if param["kind"] != "positional_or_keyword":
        out["kind"] = param["kind"]
    return out




def describe_method(name: str, func: Any, detail: str = "full") -> dict:
    entry: dict[str, Any] = {"name": name}
    summary, doc_detail = _split_doc(func)
    if summary:
        entry["summary"] = summary
    if doc_detail:
        shown = _concise_detail(doc_detail) if detail == "concise" else doc_detail
        if shown:
            entry["detail"] = shown
    param_text = _param_descriptions(doc_detail)
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
    entry["parameters"] = (
        [_concise_param(p) for p in params] if detail == "concise" else params)
    entry["billed"] = name in BILLED_METHODS
    entry["slow"] = name in SLOW_METHODS
    entry["read_only"] = name in READ_ONLY_METHODS
    entry["destructive"] = name in DESTRUCTIVE_METHODS or name in BILLED_METHODS
    return entry







def describe_api(handle: Any, detail: str = "full") -> dict:
    if handle is None:
        return {"available": False, "methods": []}
    level = detail if detail in DESCRIBE_DETAIL_LEVELS else "full"
    names = _public_method_names(handle)
    methods = []
    for name in names:
        func = getattr(handle, name, None)
        if func is None or not callable(func):
            continue
        methods.append(describe_method(name, func, detail=level))
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




            prop: dict[str, Any] = {
                "type": _json_type(param["type"]),
                "description": param.get("description") or param["type"] or "",
            }


            if prop["type"] == "array":
                items = _array_items_schema(param["type"])
                if items is not None:
                    prop["items"] = items
            properties[param["name"]] = prop
            if param["required"]:
                required.append(param["name"])
        tools.append({
            "name": f"{prefix}_{method['name']}",
            "description": _tool_description(method),
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
            "annotations": {
                "readOnlyHint": method["read_only"],
                "destructiveHint": method["destructive"],
            },


            "_meta": {"billed": method["billed"], "slow": method["slow"]},
            "call": _call_line(prefix, method),
        })
    return tools






def _tool_description(method: dict) -> str:
    summary = method.get("summary", "")
    extra = _leading_prose(method.get("detail", ""))
    description = f"{summary} {extra}".strip() if extra else summary
    if method.get("billed") and method.get("slow"):
        lead = "Billed and slow: spends the account's allowance and can run for minutes."
    elif method.get("billed"):
        lead = "Billed: can spend the account's allowance."
    elif method.get("slow"):
        lead = "Slow: can take minutes."
    else:
        return description
    cost = _section_text(method.get("detail", ""), "Cost")
    if len(cost) > _TOOL_COST_CHARS:
        cut = cost[:_TOOL_COST_CHARS]
        end = cut.rfind(". ")
        cost = cut[:end + 1] if end > 0 else cut.rstrip() + "..."
    parts = [lead, description]
    if cost:
        parts.append(f"Cost: {cost}")
    return " ".join(p for p in parts if p)




def _split_top_level(text: str, sep: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current = ""
    for char in text:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if char == sep and depth == 0:
            parts.append(current)
            current = ""
            continue
        current += char
    parts.append(current)
    return [p.strip() for p in parts if p.strip()]





def _array_items_schema(annotation: str | None) -> dict | None:
    if not annotation:
        return None
    options = [p for p in _split_top_level(str(annotation), "|")
               if p.lower() not in ("none", "nonetype")]
    if len(options) != 1:
        return None
    text = options[0]
    if text.lower().startswith("optional[") and text.endswith("]"):
        text = text[len("optional["):-1].strip()
    match = re.match(
        r"^(?:typing\.)?(?:list|tuple|sequence)\[(.*)\]$", text, re.IGNORECASE)
    if not match:
        return None
    element = _split_top_level(match.group(1), ",")
    if not element:
        return None
    kind = _json_type(element[0])
    if kind == "array":
        inner = _array_items_schema(element[0])
        return {"type": "array", "items": inner} if inner else {"type": "array"}
    return {"type": kind}









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
