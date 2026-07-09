"""Segment library catalogue: the curated set of cloud-model object prompts shown in
the before/after gallery (mirrors AI Edit's prompt presets, adapted for
segmentation).

Key difference from AI Edit: a preset's ``prompt`` is the literal cloud-model token,
a short English noun phrase that is sent to the model **unchanged in every
locale** (the cloud model's open vocabulary is English-trained). Only the ``label`` is
polyglot, so a French user reads "Bâtiment" but the box receives "building".

This module is the offline fallback catalogue. When the server catalogue is
reachable (see ``segmentation_presets_client``) it is merged on top, but the
shapes are identical so the gallery renders either source the same way. The
object set + strong/weak flags are aerial-imagery object classes phrased under
the model's "short noun phrase" rule.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QSettings

LANGS = ("en", "fr", "es", "pt")


def _p(pid: str, prompt: str, en: str, fr: str, es: str, pt: str,
       top_pick: bool = False, weak: bool = False) -> dict:
    """Build one preset. ``prompt`` is the English cloud-model token (lowercase)."""
    return {
        "id": pid,
        "prompt": prompt,
        "label": {"en": en, "fr": fr, "es": es, "pt": pt},
        "top_pick": top_pick,
        "weak": weak,
    }


def _cat(key: str, emoji: str, en: str, fr: str, es: str, pt: str,
         presets: list[dict]) -> dict:
    return {"key": key, "emoji": emoji,
            "label": {"en": en, "fr": fr, "es": es, "pt": pt},
            "presets": presets}


# Catalogue grouped by OBJECT FAMILY (what the object is), the convention used by
# aerial-imagery object datasets. Discrete countable objects are browsed by
# family, not by GIS use-domain (that taxonomy fits continuous land cover, not
# objects). All 50 prompts are kept; the few weak/continuous classes carry
# weak=True so the UI can flag them. Each category leads with one top_pick.
_CATEGORIES: list[dict] = [
    _cat("buildings", "\U0001F3E2", "Buildings & structures", "Bâtiments et structures",
         "Edificios y estructuras", "Edifícios e estruturas", [
             _p("building", "building", "Building", "Bâtiment", "Edificio", "Edifício", top_pick=True),
             _p("house", "house", "House", "Maison", "Casa", "Casa"),
             _p("rooftop", "rooftop", "Rooftop", "Toiture", "Tejado", "Telhado"),
             _p("warehouse", "warehouse", "Warehouse", "Entrepôt", "Almacén", "Galpão"),
             _p("greenhouse", "greenhouse", "Greenhouse", "Serre", "Invernadero", "Estufa"),
             _p("shed", "shed", "Shed", "Cabanon", "Cobertizo", "Galpão pequeno"),
             _p("silo", "silo", "Silo", "Silo", "Silo", "Silo"),
             _p("storage_tank", "storage tank", "Storage tank", "Réservoir",
                "Tanque de almacenamiento", "Tanque de armazenamento"),
         ]),
    _cat("vehicles_transport", "\U0001F697", "Vehicles & transport", "Véhicules et transport",
         "Vehículos y transporte", "Veículos e transporte", [
             _p("road", "road", "Road", "Route", "Carretera", "Estrada", top_pick=True),
             _p("car", "car", "Car", "Voiture", "Coche", "Carro", top_pick=True),
             _p("truck", "truck", "Truck", "Camion", "Camión", "Caminhão"),
             _p("bus", "bus", "Bus", "Bus", "Autobús", "Ônibus"),
             _p("trailer", "trailer", "Trailer", "Remorque", "Remolque", "Reboque"),
             _p("train", "train", "Train", "Train", "Tren", "Trem"),
             _p("parking_lot", "parking lot", "Parking lot", "Parking",
                "Estacionamiento", "Estacionamento"),
             _p("bridge", "bridge", "Bridge", "Pont", "Puente", "Ponte"),
             _p("roundabout", "roundabout", "Roundabout", "Rond-point", "Rotonda", "Rotatória"),
             _p("runway", "runway", "Runway", "Piste", "Pista", "Pista"),
             _p("railway_track", "railway track", "Railway track", "Voie ferrée",
                "Vía férrea", "Ferrovia"),
         ]),
    _cat("aircraft_vessels", "✈️", "Aircraft & vessels", "Aérien et maritime",
         "Aéreo y marítimo", "Aéreo e marítimo", [
             _p("airplane", "airplane", "Airplane", "Avion", "Avión", "Avião", top_pick=True),
             _p("helicopter", "helicopter", "Helicopter", "Hélicoptère",
                "Helicóptero", "Helicóptero"),
             _p("ship", "ship", "Ship", "Navire", "Barco", "Navio", top_pick=True),
             _p("boat", "boat", "Boat", "Bateau", "Bote", "Barco"),
             _p("shipping_container", "shipping container", "Shipping container",
                "Conteneur", "Contenedor", "Contêiner"),
             _p("dock", "dock", "Dock", "Quai", "Muelle", "Doca"),
         ]),
    _cat("energy_industrial", "⚡", "Energy & industrial", "Énergie et industrie",
         "Energía e industria", "Energia e indústria", [
             _p("solar_panel", "solar panel", "Solar panel", "Panneau solaire",
                "Panel solar", "Painel solar", top_pick=True),
             _p("solar_farm", "solar farm", "Solar farm", "Ferme solaire",
                "Planta solar", "Usina solar"),
             _p("wind_turbine", "wind turbine", "Wind turbine", "Éolienne",
                "Aerogenerador", "Turbina eólica"),
             _p("chimney", "chimney", "Chimney", "Cheminée", "Chimenea", "Chaminé"),
             _p("crane", "crane", "Crane", "Grue", "Grúa", "Guindaste"),
             _p("quarry", "quarry", "Quarry", "Carrière", "Cantera", "Pedreira"),
             _p("construction_site", "construction site", "Construction site",
                "Chantier", "Obra", "Canteiro de obras"),
             _p("bare_ground", "bare ground", "Bare ground", "Sol nu",
                "Suelo desnudo", "Solo exposto", weak=True),
         ]),
    _cat("sport_recreation", "\U0001F3DF️", "Sport & recreation", "Sport et loisirs",
         "Deportes y recreación", "Esportes e recreação", [
             _p("swimming_pool", "swimming pool", "Swimming pool", "Piscine",
                "Piscina", "Piscina", top_pick=True),
             _p("tennis_court", "tennis court", "Tennis court", "Court de tennis",
                "Pista de tenis", "Quadra de tênis"),
             _p("basketball_court", "basketball court", "Basketball court",
                "Terrain de basket", "Cancha de baloncesto", "Quadra de basquete"),
             _p("soccer_field", "soccer field", "Soccer field", "Terrain de foot",
                "Campo de fútbol", "Campo de futebol"),
             _p("running_track", "running track", "Running track",
                "Piste d'athlétisme", "Pista de atletismo", "Pista de atletismo"),
             _p("stadium", "stadium", "Stadium", "Stade", "Estadio", "Estádio"),
         ]),
    _cat("land_water", "\U0001F333", "Land & water", "Sol et eau",
         "Suelo y agua", "Solo e água", [
             _p("tree", "tree", "Tree", "Arbre", "Árbol", "Árvore", top_pick=True),
             _p("hedge", "hedge", "Hedge", "Haie", "Seto", "Cerca viva"),
             _p("bush", "bush", "Bush", "Buisson", "Arbusto", "Arbusto"),
             _p("vineyard", "vineyard", "Vineyard", "Vigne", "Viñedo", "Vinhedo"),
             _p("orchard", "orchard", "Orchard", "Verger", "Huerto", "Pomar"),
             _p("farm_field", "farm field", "Farm field", "Parcelle agricole",
                "Parcela agrícola", "Talhão agrícola", weak=True),
             _p("center_pivot", "center pivot", "Center pivot", "Pivot central",
                "Pivote central", "Pivô central"),
             _p("hay_bale", "hay bale", "Hay bale", "Botte de foin", "Bala de heno",
                "Fardo de feno"),
             _p("lake", "lake", "Lake", "Lac", "Lago", "Lago"),
             _p("river", "river", "River", "Rivière", "Río", "Rio", weak=True),
             _p("dam", "dam", "Dam", "Barrage", "Presa", "Barragem"),
         ]),
]

# Top-pick ids in display order (the "Popular" tab). Ordered by real Pro demand:
# buildings, vegetation, roads, water, fields, solar, then the strongest bonus
# classes (vehicles, recreation). Land
# cover (#2) has no single discrete preset so it is represented by the browse
# categories, not a Popular tile.
TOP_PICKS: list[str] = [
    "building", "tree", "road", "lake",
    "farm_field", "solar_panel", "car", "swimming_pool",
]


def current_lang() -> str:
    """Two-char locale among en/fr/es/pt, else 'en'."""
    locale = QSettings().value("locale/userLocale", "en_US") or "en"
    short = str(locale)[:2].lower()
    return short if short in LANGS else "en"


def pick_label(field, fallback: str = "") -> str:
    """Resolve a polyglot ``{en,fr,es,pt}`` label for the current locale."""
    if isinstance(field, str):
        return field
    if isinstance(field, dict):
        lang = current_lang()
        return field.get(lang) or field.get("en") or fallback
    return fallback


# Sidebar emoji by category key. Covers BOTH the offline taxonomy keys above
# and the richer server taxonomy (10 families, different keys) so the gallery
# sidebar always shows an icon, even when the server catalogue omits the emoji
# field (it currently sends emoji: null). Keep the server keys in sync with
# /api/ai-segmentation/presets.
_CATEGORY_EMOJI: dict[str, str] = {
    # offline taxonomy
    "buildings": "\U0001F3E2",               # building
    "vehicles_transport": "\U0001F697",      # car
    "aircraft_vessels": "✈️",      # airplane
    "energy_industrial": "⚡",           # high voltage
    "sport_recreation": "\U0001F3DF️",  # stadium
    "land_water": "\U0001F333",              # deciduous tree
    # server taxonomy (10 families)
    "transport": "\U0001F6E3️",         # motorway
    "vehicles": "\U0001F697",                # car
    "aircraft_maritime": "✈️",     # airplane
    "energy": "⚡",                       # high voltage
    "water": "\U0001F4A7",                   # droplet
    "vegetation": "\U0001F333",              # deciduous tree
    "agriculture": "\U0001F33E",             # sheaf of rice
    "sports": "\U0001F3DF️",            # stadium
    "land": "\U0001F3D4️",              # snow-capped mountain
}
_CATEGORY_EMOJI_DEFAULT = "\U0001F4C2"       # open folder: generic category icon


def category_emoji(key: str) -> str:
    """Sidebar emoji for a category key, falling back to a generic folder glyph
    so an unknown/new server key still renders an icon."""
    return _CATEGORY_EMOJI.get(key, _CATEGORY_EMOJI_DEFAULT)


def fallback_categories() -> list[dict]:
    """The offline catalogue (ordered domains, each with its presets)."""
    return _CATEGORIES


def all_presets() -> list[dict]:
    return [p for cat in _CATEGORIES for p in cat["presets"]]


# Built-in synonym map for the zero-result assist: a failed
# prompt maps to a few alternate English cloud-model tokens the user can retry with one
# click. Ships in the plugin (the server presets payload carries no synonym
# field yet); keys are lowercase, high-signal, kept small.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "house": ("building", "rooftop"),
    "field": ("farmland", "crop field"),
    "forest": ("tree", "woodland"),
    "car": ("vehicle",),
    "lake": ("water",),
    "path": ("road", "track"),
    "panel": ("solar panel",),
    "parking": ("parking lot",),
    "boat": ("vessel",),
    "grass": ("vegetation",),
}


def synonyms_for(prompt: str) -> list[str]:
    """Up to 3 alternate English tokens for a failed prompt, or [] if none.

    Case-insensitive; matches the whole prompt first, then its first word, then
    a naive singular of that word (so 'houses' / 'a house' still map)."""
    if not prompt:
        return []
    key = prompt.strip().lower()
    alts = _SYNONYMS.get(key)
    if not alts:
        words = key.split()
        first = words[0] if words else ""
        alts = _SYNONYMS.get(first) or _SYNONYMS.get(first.rstrip("s"))
    return list(alts[:3]) if alts else []


def known_tokens() -> list[str]:
    """Flat, de-duplicated English prompt tokens (for the validator's
    'did you mean' suggestions)."""
    seen: dict[str, None] = {}
    for p in all_presets():
        seen.setdefault(p["prompt"], None)
    return list(seen.keys())


def _fold_label(text) -> str:
    """Accent-fold a label to lowercase ASCII ('Bâtiment' -> 'batiment')."""
    import unicodedata
    return (
        unicodedata.normalize("NFKD", str(text or ""))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
        .strip()
    )


def token_by_localized_label() -> dict[str, str]:
    """Accent-folded localized preset label (en/fr/es/pt) -> English prompt
    token, built from the live catalogue (the cached server one when
    available, the offline fallback otherwise).

    This is the scale lever behind the prompt box's silent translation: a
    user can type the object in their own language ("piscine") and the run
    sends the English cloud-model token ("swimming pool"). Every label the library
    ships - in every language it ships - is accepted automatically, with no
    hand-maintained dictionary to grow stale.
    """
    cats: list[dict] = []
    try:
        # Lazy import: the client module imports this one for its fallbacks.
        from .segmentation_presets_client import cached_or_offline_catalog
        cats, _top = cached_or_offline_catalog()
    except Exception:  # noqa: BLE001 -- offline fallback below
        cats = []
    index: dict[str, str] = {}
    for cat in cats or _CATEGORIES:
        for p in cat.get("presets", []):
            token = (p.get("prompt") or "").strip()
            if not token:
                continue
            label = p.get("label")
            values = list(label.values()) if isinstance(label, dict) else [label]
            for value in values:
                folded = _fold_label(value)
                if folded:
                    index.setdefault(folded, token)
    return index
