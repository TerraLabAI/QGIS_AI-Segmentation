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

LANGS = ("en", "fr", "es", "pt", "de", "it", "nl", "pl", "id", "ja", "zh_CN", "zh_TW")


def _p(pid: str, prompt: str, en: str, fr: str, es: str, pt: str, top_pick: bool = False, weak: bool = False) -> dict:
    """Build one preset. ``prompt`` is the English cloud-model token (lowercase)."""
    return {
        "id": pid,
        "prompt": prompt,
        "label": {"en": en, "fr": fr, "es": es, "pt": pt, **_PRESET_L10N.get(pid, {})},
        "top_pick": top_pick,
        "weak": weak,
    }


def _cat(key: str, emoji: str, en: str, fr: str, es: str, pt: str, presets: list[dict]) -> dict:
    return {
        "key": key,
        "emoji": emoji,
        "label": {"en": en, "fr": fr, "es": es, "pt": pt, **_CAT_L10N.get(key, {})},
        "presets": presets,
    }


# Extra-locale label overlays (de/it/nl/pl/id/ja/zh_CN/zh_TW), merged on top of the
# inline en/fr/es/pt labels by ``_p``/``_cat``. The cloud-model ``prompt`` token stays
# English in every locale; only the browsed label follows the UI language. A missing
# entry falls back to English via ``pick_label``, so this table need not be exhaustive.
_L10N_LANGS = ("de", "it", "nl", "pl", "id", "ja", "zh_CN", "zh_TW")


def _l(de: str, it: str, nl: str, pl: str, id_: str, ja: str, zh_cn: str, zh_tw: str) -> dict:
    return dict(zip(_L10N_LANGS, (de, it, nl, pl, id_, ja, zh_cn, zh_tw)))


_CAT_L10N: dict[str, dict] = {
    "buildings": _l(
        "Gebäude & Strukturen",
        "Edifici e strutture",
        "Gebouwen & structuren",
        "Budynki i struktury",
        "Bangunan & struktur",
        "建物・構造物",
        "建筑与结构",
        "建築與結構",
    ),
    "vehicles_transport": _l(
        "Fahrzeuge & Transport",
        "Veicoli e trasporti",
        "Voertuigen & transport",
        "Pojazdy i transport",
        "Kendaraan & transportasi",
        "車両・輸送",
        "车辆与运输",
        "車輛與運輸",
    ),
    "aircraft_vessels": _l(
        "Luftfahrzeuge & Schiffe",
        "Aeromobili e navi",
        "Vliegtuigen & vaartuigen",
        "Samoloty i statki",
        "Pesawat & kapal",
        "航空機・船舶",
        "飞行器与船舶",
        "飛行器與船隻",
    ),
    "energy_industrial": _l(
        "Energie & Industrie",
        "Energia e industria",
        "Energie & industrie",
        "Energia i przemysł",
        "Energi & industri",
        "エネルギー・産業",
        "能源与工业",
        "能源與工業",
    ),
    "sport_recreation": _l(
        "Sport & Freizeit",
        "Sport e ricreazione",
        "Sport & recreatie",
        "Sport i rekreacja",
        "Olahraga & rekreasi",
        "スポーツ・レクリエーション",
        "运动与娱乐",
        "運動與休閒",
    ),
    "land_water": _l(
        "Land & Wasser",
        "Terreno e acqua",
        "Land & water",
        "Ląd i woda",
        "Lahan & air",
        "土地・水域",
        "土地与水体",
        "陸地與水體",
    ),
}

_PRESET_L10N: dict[str, dict] = {
    "building": _l("Gebäude", "Edificio", "Gebouw", "Budynek", "Bangunan", "建物", "建筑", "建築"),
    "house": _l("Haus", "Casa", "Huis", "Dom", "Rumah", "住宅", "房屋", "房屋"),
    "rooftop": _l("Dach", "Tetto", "Dak", "Dach", "Atap", "屋根", "屋顶", "屋頂"),
    "warehouse": _l("Lagerhaus", "Magazzino", "Pakhuis", "Magazyn", "Gudang", "倉庫", "仓库", "倉庫"),
    "greenhouse": _l("Gewächshaus", "Serra", "Kas", "Szklarnia", "Rumah kaca", "温室", "温室", "溫室"),
    "shed": _l("Schuppen", "Capannone", "Schuur", "Szopa", "Gubuk", "小屋", "棚屋", "棚舍"),
    "silo": _l("Silo", "Silo", "Silo", "Silo", "Silo", "サイロ", "筒仓", "穀倉"),
    "storage_tank": _l(
        "Speichertank", "Serbatoio", "Opslagtank", "Zbiornik", "Tangki penyimpanan", "タンク", "储罐", "儲存槽"
    ),
    "road": _l("Straße", "Strada", "Weg", "Droga", "Jalan", "道路", "道路", "道路"),
    "car": _l("Auto", "Auto", "Auto", "Samochód", "Mobil", "車", "汽车", "汽車"),
    "truck": _l("Lastwagen", "Camion", "Vrachtwagen", "Ciężarówka", "Truk", "トラック", "卡车", "卡車"),
    "bus": _l("Bus", "Autobus", "Bus", "Autobus", "Bus", "バス", "公交车", "公車"),
    "trailer": _l("Anhänger", "Rimorchio", "Aanhangwagen", "Przyczepa", "Trailer", "トレーラー", "拖车", "拖車"),
    "train": _l("Zug", "Treno", "Trein", "Pociąg", "Kereta", "列車", "火车", "火車"),
    "parking_lot": _l(
        "Parkplatz", "Parcheggio", "Parkeerplaats", "Parking", "Tempat parkir", "駐車場", "停车场", "停車場"
    ),
    "bridge": _l("Brücke", "Ponte", "Brug", "Most", "Jembatan", "橋", "桥梁", "橋"),
    "roundabout": _l("Kreisverkehr", "Rotatoria", "Rotonde", "Rondo", "Bundaran", "ロータリー", "环岛", "圓環"),
    "runway": _l("Landebahn", "Pista", "Startbaan", "Pas startowy", "Landasan pacu", "滑走路", "跑道", "跑道"),
    "railway_track": _l(
        "Eisenbahnstrecke", "Binario", "Spoorweg", "Tor kolejowy", "Rel kereta", "線路", "铁轨", "鐵軌"
    ),
    "airplane": _l("Flugzeug", "Aereo", "Vliegtuig", "Samolot", "Pesawat", "飛行機", "飞机", "飛機"),
    "helicopter": _l(
        "Hubschrauber", "Elicottero", "Helikopter", "Helikopter", "Helikopter", "ヘリコプター", "直升机", "直升機"
    ),
    "ship": _l("Schiff", "Nave", "Schip", "Statek", "Kapal", "船舶", "船舶", "船舶"),
    "boat": _l("Boot", "Barca", "Boot", "Łódź", "Perahu", "ボート", "小船", "小船"),
    "shipping_container": _l(
        "Container", "Contenitore", "Zeecontainer", "Kontener", "Kontainer", "コンテナ", "集装箱", "貨櫃"
    ),
    "dock": _l("Kai", "Molo", "Dock", "Dok", "Dermaga", "埠頭", "码头", "碼頭"),
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
    "solar_farm": _l(
        "Solaranlage",
        "Parco solare",
        "Zonneboerderij",
        "Farma słoneczna",
        "Ladang surya",
        "太陽光発電所",
        "太阳能电站",
        "太陽能農場",
    ),
    "wind_turbine": _l(
        "Windkraftanlage",
        "Turbina eolica",
        "Windmolen",
        "Turbina wiatrowa",
        "Turbin angin",
        "風力発電機",
        "风力发电机",
        "風力發電機",
    ),
    "chimney": _l("Schornstein", "Camino", "Schoorsteen", "Komin", "Cerobong", "煙突", "烟囱", "煙囪"),
    "crane": _l("Kran", "Gru", "Kraan", "Żuraw", "Derek", "クレーン", "起重机", "吊車"),
    "quarry": _l("Steinbruch", "Cava", "Groeve", "Kamieniołom", "Tambang terbuka", "採石場", "采石场", "採石場"),
    "construction_site": _l(
        "Baustelle", "Cantiere", "Bouwplaats", "Plac budowy", "Lokasi konstruksi", "建設現場", "建筑工地", "工地"
    ),
    "bare_ground": _l(
        "Freifläche", "Suolo nudo", "Kale grond", "Gołe podłoże", "Tanah terbuka", "裸地", "裸地", "裸露地面"
    ),
    "swimming_pool": _l("Schwimmbecken", "Piscina", "Zwembad", "Basen", "Kolam renang", "プール", "游泳池", "游泳池"),
    "tennis_court": _l(
        "Tennisplatz",
        "Campo da tennis",
        "Tennisbaan",
        "Kort tenisowy",
        "Lapangan tenis",
        "テニスコート",
        "网球场",
        "網球場",
    ),
    "basketball_court": _l(
        "Basketballplatz",
        "Campo da basket",
        "Basketbalveld",
        "Boisko koszykówki",
        "Lapangan basket",
        "バスケットボールコート",
        "篮球场",
        "籃球場",
    ),
    "soccer_field": _l(
        "Fußballplatz",
        "Campo da calcio",
        "Voetbalveld",
        "Boisko piłkarskie",
        "Lapangan sepak bola",
        "サッカー場",
        "足球场",
        "足球場",
    ),
    "running_track": _l(
        "Laufbahn", "Pista", "Hardloopbaan", "Bieżnia", "Lintasan lari", "トラック", "田径跑道", "田徑道"
    ),
    "stadium": _l("Stadion", "Stadio", "Stadion", "Stadion", "Stadion", "スタジアム", "体育场", "體育場"),
    "tree": _l("Baum", "Albero", "Boom", "Drzewo", "Pohon", "樹木", "树木", "樹木"),
    "hedge": _l("Hecke", "Siepe", "Haag", "Żywopłot", "Pagar hidup", "生垣", "树篱", "樹籬"),
    "bush": _l("Strauch", "Cespuglio", "Struik", "Krzak", "Semak", "灌木", "灌木", "灌木"),
    "vineyard": _l("Weinberg", "Vigneto", "Wijngaard", "Winnica", "Kebun anggur", "ぶどう畑", "葡萄园", "葡萄園"),
    "orchard": _l("Obstgarten", "Frutteto", "Boomgaard", "Sad", "Kebun buah", "果樹園", "果园", "果園"),
    "farm_field": _l("Acker", "Campo agricolo", "Akker", "Pole uprawne", "Lahan pertanian", "農地", "农田", "農田"),
    "center_pivot": _l(
        "Bewässerungsanlage",
        "Irrigazione pivot",
        "Beregeningsinstallatie",
        "System nawodnień",
        "Pusat pivot",
        "センターピボット",
        "中心支轴灌溉",
        "圓形噴灌機",
    ),
    "hay_bale": _l(
        "Heuballen", "Balla di fieno", "Hooibaal", "Bała siana", "Bal jerami", "牧草ロール", "干草捆", "草捆"
    ),
    "lake": _l("See", "Lago", "Meer", "Jezioro", "Danau", "湖", "湖泊", "湖泊"),
    "river": _l("Fluss", "Fiume", "Rivier", "Rzeka", "Sungai", "河川", "河流", "河川"),
    "dam": _l("Damm", "Diga", "Dam", "Zapora", "Bendungan", "ダム", "大坝", "水壩"),
}


# Catalogue grouped by OBJECT FAMILY (what the object is), the convention used by
# aerial-imagery object datasets. Discrete countable objects are browsed by
# family, not by GIS use-domain (that taxonomy fits continuous land cover, not
# objects). All 50 prompts are kept; the few weak/continuous classes carry
# weak=True so the UI can flag them. Each category leads with one top_pick.
_CATEGORIES: list[dict] = [
    _cat(
        "buildings",
        "\U0001f3e2",
        "Buildings & structures",
        "Bâtiments et structures",
        "Edificios y estructuras",
        "Edifícios e estruturas",
        [
            _p("building", "building", "Building", "Bâtiment", "Edificio", "Edifício", top_pick=True),
            _p("house", "house", "House", "Maison", "Casa", "Casa"),
            _p("rooftop", "rooftop", "Rooftop", "Toiture", "Tejado", "Telhado"),
            _p("warehouse", "warehouse", "Warehouse", "Entrepôt", "Almacén", "Galpão"),
            _p("greenhouse", "greenhouse", "Greenhouse", "Serre", "Invernadero", "Estufa"),
            _p("shed", "shed", "Shed", "Cabanon", "Cobertizo", "Galpão pequeno"),
            _p("silo", "silo", "Silo", "Silo", "Silo", "Silo"),
            _p(
                "storage_tank",
                "storage tank",
                "Storage tank",
                "Réservoir",
                "Tanque de almacenamiento",
                "Tanque de armazenamento",
            ),
        ],
    ),
    _cat(
        "vehicles_transport",
        "\U0001f697",
        "Vehicles & transport",
        "Véhicules et transport",
        "Vehículos y transporte",
        "Veículos e transporte",
        [
            _p("road", "road", "Road", "Route", "Carretera", "Estrada", top_pick=True),
            _p("car", "car", "Car", "Voiture", "Coche", "Carro", top_pick=True),
            _p("truck", "truck", "Truck", "Camion", "Camión", "Caminhão"),
            _p("bus", "bus", "Bus", "Bus", "Autobús", "Ônibus"),
            _p("trailer", "trailer", "Trailer", "Remorque", "Remolque", "Reboque"),
            _p("train", "train", "Train", "Train", "Tren", "Trem"),
            _p("parking_lot", "parking lot", "Parking lot", "Parking", "Estacionamiento", "Estacionamento"),
            _p("bridge", "bridge", "Bridge", "Pont", "Puente", "Ponte"),
            _p("roundabout", "roundabout", "Roundabout", "Rond-point", "Rotonda", "Rotatória"),
            _p("runway", "runway", "Runway", "Piste", "Pista", "Pista"),
            _p("railway_track", "railway track", "Railway track", "Voie ferrée", "Vía férrea", "Ferrovia"),
        ],
    ),
    _cat(
        "aircraft_vessels",
        "✈️",
        "Aircraft & vessels",
        "Aérien et maritime",
        "Aéreo y marítimo",
        "Aéreo e marítimo",
        [
            _p("airplane", "airplane", "Airplane", "Avion", "Avión", "Avião", top_pick=True),
            _p("helicopter", "helicopter", "Helicopter", "Hélicoptère", "Helicóptero", "Helicóptero"),
            _p("ship", "ship", "Ship", "Navire", "Barco", "Navio", top_pick=True),
            _p("boat", "boat", "Boat", "Bateau", "Bote", "Barco"),
            _p(
                "shipping_container", "shipping container", "Shipping container", "Conteneur", "Contenedor", "Contêiner"
            ),
            _p("dock", "dock", "Dock", "Quai", "Muelle", "Doca"),
        ],
    ),
    _cat(
        "energy_industrial",
        "⚡",
        "Energy & industrial",
        "Énergie et industrie",
        "Energía e industria",
        "Energia e indústria",
        [
            _p(
                "solar_panel",
                "solar panel",
                "Solar panel",
                "Panneau solaire",
                "Panel solar",
                "Painel solar",
                top_pick=True,
            ),
            _p("solar_farm", "solar farm", "Solar farm", "Ferme solaire", "Planta solar", "Usina solar"),
            _p("wind_turbine", "wind turbine", "Wind turbine", "Éolienne", "Aerogenerador", "Turbina eólica"),
            _p("chimney", "chimney", "Chimney", "Cheminée", "Chimenea", "Chaminé"),
            _p("crane", "crane", "Crane", "Grue", "Grúa", "Guindaste"),
            _p("quarry", "quarry", "Quarry", "Carrière", "Cantera", "Pedreira"),
            _p("construction_site", "construction site", "Construction site", "Chantier", "Obra", "Canteiro de obras"),
            _p("bare_ground", "bare ground", "Bare ground", "Sol nu", "Suelo desnudo", "Solo exposto", weak=True),
        ],
    ),
    _cat(
        "sport_recreation",
        "\U0001f3df️",
        "Sport & recreation",
        "Sport et loisirs",
        "Deportes y recreación",
        "Esportes e recreação",
        [
            _p("swimming_pool", "swimming pool", "Swimming pool", "Piscine", "Piscina", "Piscina", top_pick=True),
            _p("tennis_court", "tennis court", "Tennis court", "Court de tennis", "Pista de tenis", "Quadra de tênis"),
            _p(
                "basketball_court",
                "basketball court",
                "Basketball court",
                "Terrain de basket",
                "Cancha de baloncesto",
                "Quadra de basquete",
            ),
            _p(
                "soccer_field", "soccer field", "Soccer field", "Terrain de foot", "Campo de fútbol", "Campo de futebol"
            ),
            _p(
                "running_track",
                "running track",
                "Running track",
                "Piste d'athlétisme",
                "Pista de atletismo",
                "Pista de atletismo",
            ),
            _p("stadium", "stadium", "Stadium", "Stade", "Estadio", "Estádio"),
        ],
    ),
    _cat(
        "land_water",
        "\U0001f333",
        "Land & water",
        "Sol et eau",
        "Suelo y agua",
        "Solo e água",
        [
            _p("tree", "tree", "Tree", "Arbre", "Árbol", "Árvore", top_pick=True),
            _p("hedge", "hedge", "Hedge", "Haie", "Seto", "Cerca viva"),
            _p("bush", "bush", "Bush", "Buisson", "Arbusto", "Arbusto"),
            _p("vineyard", "vineyard", "Vineyard", "Vigne", "Viñedo", "Vinhedo"),
            _p("orchard", "orchard", "Orchard", "Verger", "Huerto", "Pomar"),
            _p(
                "farm_field",
                "farm field",
                "Farm field",
                "Parcelle agricole",
                "Parcela agrícola",
                "Talhão agrícola",
                weak=True,
            ),
            _p("center_pivot", "center pivot", "Center pivot", "Pivot central", "Pivote central", "Pivô central"),
            _p("hay_bale", "hay bale", "Hay bale", "Botte de foin", "Bala de heno", "Fardo de feno"),
            _p("lake", "lake", "Lake", "Lac", "Lago", "Lago"),
            _p("river", "river", "River", "Rivière", "Río", "Rio", weak=True),
            _p("dam", "dam", "Dam", "Barrage", "Presa", "Barragem"),
        ],
    ),
]

# Top-pick ids in display order (the "Popular" tab). Ordered by real Pro demand:
# buildings, vegetation, roads, water, fields, solar, then the strongest bonus
# classes (vehicles, recreation). Land
# cover (#2) has no single discrete preset so it is represented by the browse
# categories, not a Popular tile.
TOP_PICKS: list[str] = [
    "building",
    "tree",
    "road",
    "lake",
    "farm_field",
    "solar_panel",
    "car",
    "swimming_pool",
]


def current_lang() -> str:
    """Resolve the QGIS UI locale to one of the catalogue languages, else 'en'.

    Chinese needs the region subtag (Taiwan/Hong Kong/Hant carry Traditional)
    because a bare two-char slice would collapse ``zh_CN``/``zh_TW`` to ``zh``,
    which is not a catalogue key and would fall back to English.
    """
    locale = str(QSettings().value("locale/userLocale", "en_US") or "en")
    norm = locale.replace("-", "_").lower()
    short = norm[:2]
    if short == "zh":
        if any(tag in norm for tag in ("tw", "hk", "hant", "mo")):
            return "zh_TW"
        return "zh_CN"
    return short if short in LANGS else "en"


def pick_label(field, fallback: str = "") -> str:
    """Resolve a polyglot label dict for the current locale."""
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
    "buildings": "\U0001f3e2",  # building
    "vehicles_transport": "\U0001f697",  # car
    "aircraft_vessels": "✈️",  # airplane
    "energy_industrial": "⚡",  # high voltage
    "sport_recreation": "\U0001f3df️",  # stadium
    "land_water": "\U0001f333",  # deciduous tree
    # server taxonomy (10 families)
    "transport": "\U0001f6e3️",  # motorway
    "vehicles": "\U0001f697",  # car
    "aircraft_maritime": "✈️",  # airplane
    "energy": "⚡",  # high voltage
    "water": "\U0001f4a7",  # droplet
    "vegetation": "\U0001f333",  # deciduous tree
    "agriculture": "\U0001f33e",  # sheaf of rice
    "sports": "\U0001f3df️",  # stadium
    "land": "\U0001f3d4️",  # snow-capped mountain
}
_CATEGORY_EMOJI_DEFAULT = "\U0001f4c2"  # open folder: generic category icon


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

    return unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii").lower().strip()


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
