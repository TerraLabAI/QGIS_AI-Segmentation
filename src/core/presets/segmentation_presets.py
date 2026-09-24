


























from __future__ import annotations

from typing import Any, Iterable

from qgis.PyQt.QtCore import QSettings

from .segmentation_search_terms import preset_search_terms

LANGS = ("en", "fr", "es", "pt", "de", "it", "nl", "pl", "id", "ja", "zh_CN", "zh_TW")









TOP_PICKS: list[str] = [
    "building",
    "house",
    "tree",
    "road",
    "water",
    "car",
    "parking_lot",
    "solar_panel",
    "swimming_pool",
]


def _p(pid: str, prompt: str, en: str, fr: str, es: str, pt: str, *, weak: bool = False) -> dict:

    return {
        "id": pid,
        "prompt": prompt,
        "label": {"en": en, "fr": fr, "es": es, "pt": pt, **_PRESET_L10N.get(pid, {})},
        "top_pick": pid in TOP_PICKS,
        "weak": weak,
        "search_terms": preset_search_terms(pid),
    }


def _cat(key: str, en: str, fr: str, es: str, pt: str, presets: list[dict]) -> dict:
    return {
        "key": key,
        "emoji": category_emoji(key),
        "label": {"en": en, "fr": fr, "es": es, "pt": pt, **_CAT_L10N.get(key, {})},
        "presets": presets,
    }






_L10N_LANGS = ("de", "it", "nl", "pl", "id", "ja", "zh_CN", "zh_TW")


def _l(de: str, it: str, nl: str, pl: str, id_: str, ja: str, zh_cn: str, zh_tw: str) -> dict:
    return dict(zip(_L10N_LANGS, (de, it, nl, pl, id_, ja, zh_cn, zh_tw)))




_CAT_L10N: dict[str, dict] = {
    "buildings": _l("Gebäude und Dächer", "Edifici e tetti", "Gebouwen en daken", "Budynki i dachy",
                    "Bangunan dan atap", "建物・屋根", "建筑与屋顶", "建築與屋頂"),
    "vegetation": _l("Bäume und Vegetation", "Alberi e vegetazione", "Bomen en begroeiing",
                     "Drzewa i roślinność", "Pohon dan vegetasi", "樹木・植生", "树木与植被", "樹木與植被"),
    "transport": _l("Straßen und Infrastruktur", "Strade e infrastrutture", "Wegen en infrastructuur",
                    "Drogi i infrastruktura", "Jalan dan infrastruktur", "道路・インフラ",
                    "道路与基础设施", "道路與基礎設施"),
    "land_water": _l("Wasser und Boden", "Acqua e suolo", "Water en bodem", "Woda i grunt",
                     "Air dan lahan", "水域・地面", "水体与地面", "水體與地面"),
    "vehicles_transport": _l("Fahrzeuge", "Veicoli", "Voertuigen", "Pojazdy",
                             "Kendaraan", "車両", "车辆", "車輛"),
    "agriculture": _l("Felder und Kulturen", "Parcelle e colture", "Percelen en gewassen",
                      "Działki i uprawy", "Lahan dan tanaman", "農地・作物", "地块与作物", "地塊與作物"),
    "energy": _l("Solar- und Windenergie", "Energia solare ed eolica", "Zonne- en windenergie",
                 "Energia słoneczna i wiatrowa", "Energi surya dan angin", "太陽光・風力",
                 "太阳能与风能", "太陽能與風能"),
    "sport_recreation": _l("Sport und Freizeit", "Sport e tempo libero", "Sport en recreatie",
                           "Sport i rekreacja", "Olahraga dan rekreasi", "スポーツ・レジャー",
                           "运动与休闲", "運動與休閒"),
    "aircraft_vessels": _l("Flugzeuge und Boote", "Aerei e barche", "Vliegtuigen en boten",
                           "Samoloty i łodzie", "Pesawat dan kapal", "航空機・船舶", "飞机与船只", "飛機與船隻"),
    "industry": _l("Industrie und Baustellen", "Industria e cantieri", "Industrie en werken",
                   "Przemysł i budowy", "Industri dan konstruksi", "工業・工事", "工业与施工", "工業與施工"),
}



_PRESET_L10N: dict[str, dict] = {
    "building": _l("Gebäude", "Edificio", "Gebouw", "Budynek", "Bangunan", "建物", "建筑", "建築"),
    "house": _l("Haus", "Casa", "Huis", "Dom", "Rumah", "住宅", "房屋", "房屋"),
    "road": _l("Straße", "Strada", "Weg", "Droga", "Jalan", "道路", "道路", "道路"),
    "car": _l("Auto", "Auto", "Auto", "Samochód", "Mobil", "車", "汽车", "汽車"),
    "parking_lot": _l(
        "Parkplatz", "Parcheggio", "Parkeerplaats", "Parking", "Tempat parkir", "駐車場", "停车场", "停車場"
    ),
    "solar_panel": _l(
        "Solarpanel",
        "Pannello solare",
        "Zonnepaneel",
        "Panel słoneczny",
        "Panel surya",
        "太陽光パネル",
        "太阳能板",
        "太陽能板",
    ),
    "swimming_pool": _l("Schwimmbecken", "Piscina", "Zwembad", "Basen", "Kolam renang", "プール", "游泳池", "游泳池"),
    "tree": _l("Baum", "Albero", "Boom", "Drzewo", "Pohon", "樹木", "树木", "樹木"),
}







_CATEGORY_EMOJI: dict[str, str] = {

    "buildings": "\U0001f3e2",
    "vegetation": "\U0001f333",
    "transport": "\U0001f6e3️",
    "land_water": "\U0001f4a7",
    "vehicles_transport": "\U0001f697",
    "agriculture": "\U0001f33e",
    "energy": "⚡",
    "sport_recreation": "\U0001f3df️",
    "aircraft_vessels": "✈️",
    "industry": "\U0001f3ed",

    "energy_industrial": "⚡",
    "vehicles": "\U0001f697",
    "aircraft_maritime": "✈️",
    "water": "\U0001f4a7",
    "sports": "\U0001f3df️",
    "land": "\U0001f3d4️",
}
_CATEGORY_EMOJI_DEFAULT = "\U0001f4c2"


def category_emoji(key: str) -> str:


    return _CATEGORY_EMOJI.get(key, _CATEGORY_EMOJI_DEFAULT)










_CATEGORIES: list[dict] = [
    _cat(
        "buildings",
        "Buildings and rooftops",
        "Bâtiments et toitures",
        "Edificios y tejados",
        "Edifícios e telhados",
        [
            _p("building", "building", "Building", "Bâtiment", "Edificio", "Edifício"),
            _p("house", "house", "House", "Maison", "Casa", "Casa"),
        ],
    ),
    _cat(
        "vegetation",
        "Trees and vegetation",
        "Arbres et végétation",
        "Árboles y vegetación",
        "Árvores e vegetação",
        [
            _p("tree", "tree", "Tree", "Arbre", "Árbol", "Árvore"),
        ],
    ),
    _cat(
        "transport",
        "Roads and infrastructure",
        "Routes et infrastructures",
        "Carreteras e infraestructuras",
        "Estradas e infraestrutura",
        [
            _p("road", "road", "Road", "Route", "Carretera", "Estrada"),
            _p("parking_lot", "parking lot", "Parking lot", "Parking", "Estacionamiento", "Estacionamento"),
        ],
    ),
    _cat(
        "land_water",
        "Water and land",
        "Eau et sols",
        "Agua y suelo",
        "Água e solo",
        [
            _p("water", "water", "Water", "Eau", "Agua", "Água", weak=True),
        ],
    ),
    _cat(
        "vehicles_transport",
        "Vehicles",
        "Véhicules",
        "Vehículos",
        "Veículos",
        [
            _p("car", "car", "Car", "Voiture", "Coche", "Carro"),
        ],
    ),
    _cat(
        "agriculture",
        "Fields and crops",
        "Parcelles et cultures",
        "Parcelas y cultivos",
        "Talhões e culturas",
        [],
    ),
    _cat(
        "energy",
        "Solar and wind energy",
        "Énergie solaire et éolienne",
        "Energía solar y eólica",
        "Energia solar e eólica",
        [
            _p("solar_panel", "solar panel", "Solar panel", "Panneau solaire", "Panel solar", "Painel solar"),
        ],
    ),
    _cat(
        "sport_recreation",
        "Sport and leisure",
        "Sport et loisirs",
        "Deporte y ocio",
        "Esporte e lazer",
        [
            _p("swimming_pool", "swimming pool", "Swimming pool", "Piscine", "Piscina", "Piscina"),
        ],
    ),
    _cat(
        "aircraft_vessels",
        "Aircraft and boats",
        "Avions et bateaux",
        "Aviones y barcos",
        "Aviões e barcos",
        [],
    ),
    _cat(
        "industry",
        "Industry and works",
        "Industrie et chantiers",
        "Industria y obras",
        "Indústria e obras",
        [],
    ),
]


def current_lang() -> str:






    locale = str(QSettings().value("locale/userLocale", "en_US") or "en")
    norm = locale.replace("-", "_").lower()
    short = norm[:2]
    if short == "zh":
        if any(tag in norm for tag in ("tw", "hk", "hant", "mo")):
            return "zh_TW"
        return "zh_CN"
    return short if short in LANGS else "en"


def pick_label(field, fallback: str = "") -> str:

    if isinstance(field, str):
        return field
    if isinstance(field, dict):
        lang = current_lang()
        return field.get(lang) or field.get("en") or fallback
    return fallback


def fallback_categories() -> list[dict]:






    return _FALLBACK_CATEGORIES



_FALLBACK_CATEGORIES: list[dict] = [cat for cat in _CATEGORIES if cat["presets"]]


def catalog_revision() -> str:







    try:



        from .segmentation_presets_client import _CACHE_TS_KEY

        return str(QSettings().value(_CACHE_TS_KEY, "") or "")
    except Exception:  # noqa: BLE001
        return ""


def merged_categories(served: list[dict] | None) -> list[dict]:

















    if not served:
        return fallback_categories()
    known: set[str] = set()
    out: list[dict] = []
    for category in served:
        if not isinstance(category, dict):
            continue
        presets = category.get("presets")
        if presets is None:
            presets = []
        if not isinstance(presets, list):
            continue
        clean = [preset for preset in presets if isinstance(preset, dict)]


        out.append(dict(category, presets=clean))
        for preset in clean:
            token = str(preset.get("prompt") or "").strip().lower()
            if token:
                known.add(token)
    if not out:
        return fallback_categories()
    by_key = {str(cat.get("key") or ""): cat for cat in out}
    for category in _CATEGORIES:
        missing = [
            preset for preset in category["presets"]
            if str(preset.get("prompt") or "").strip().lower() not in known
        ]
        if not missing:
            continue
        target = by_key.get(category["key"])
        if target is None:
            out.append(dict(category, presets=list(missing)))
            by_key[category["key"]] = out[-1]
            continue


        target["presets"] = list(target.get("presets") or []) + missing
    return out


def known_tokens() -> list[str]:










    cats: list[dict] = []
    try:

        from .segmentation_presets_client import cached_or_offline_catalog

        cats, _top = cached_or_offline_catalog()
    except Exception:  # noqa: BLE001
        cats = []
    seen: dict[str, None] = {}
    for cat in merged_categories(cats):
        for preset in cat.get("presets", []) or []:
            if not isinstance(preset, dict):
                continue
            token = str(preset.get("prompt") or "").strip()
            if token:
                seen.setdefault(token, None)
    return list(seen.keys())




_LETTERS_THAT_DO_NOT_DECOMPOSE = str.maketrans(
    {"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D",
     "ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE"}
)


def fold_search_text(text) -> str:






    import unicodedata

    folded = str(text or "").translate(_LETTERS_THAT_DO_NOT_DECOMPOSE)
    decomposed = unicodedata.normalize("NFKD", folded)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower().strip()


def search_terms_of(preset) -> list[str]:






    if not isinstance(preset, dict):
        return []
    terms = preset.get("search_terms")
    values: Iterable[Any]
    if isinstance(terms, dict):
        values = terms.values()
    elif isinstance(terms, (list, tuple)):
        values = terms
    elif terms:
        values = [terms]
    else:
        return []
    out: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            out.extend(str(v) for v in value if v)
        elif value:
            out.append(str(value))
    return out


def preset_search_haystack(preset, category_label: str = "") -> str:








    labels = (preset or {}).get("label")
    parts = [str((preset or {}).get("prompt", "")), category_label]
    if isinstance(labels, dict):
        parts.extend(str(v) for v in labels.values())
    else:
        parts.append(pick_label(labels, ""))
    parts.extend(search_terms_of(preset))
    return fold_search_text(" ".join(p for p in parts if p))


def preset_matches_query(preset, query: str, category_label: str = "") -> bool:


    folded = fold_search_text(query)
    return bool(folded) and folded in preset_search_haystack(preset, category_label)


def token_by_localized_label() -> dict[str, str]:










    cats: list[dict] = []
    try:

        from .segmentation_presets_client import cached_or_offline_catalog

        cats, _top = cached_or_offline_catalog()
    except Exception:  # noqa: BLE001
        cats = []
    index: dict[str, str] = {}
    for cat in merged_categories(cats):
        for p in cat.get("presets", []) or []:
            if not isinstance(p, dict):
                continue
            token = (p.get("prompt") or "").strip()
            if not token:
                continue
            label = p.get("label")
            values = list(label.values()) if isinstance(label, dict) else [label]
            for value in values:
                folded = fold_search_text(value)
                if folded:
                    index.setdefault(folded, token)
    return index
