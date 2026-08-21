"""Segment library catalogue: the curated set of cloud-model object prompts shown in
the before/after gallery (mirrors AI Edit's prompt presets, adapted for
segmentation).

Key difference from AI Edit: a preset's ``prompt`` is the literal cloud-model token,
a short English noun phrase that is sent to the model **unchanged in every
locale** (the cloud model's open vocabulary is English-trained). Only the ``label``
and the optional ``search_terms`` are polyglot, so a French user reads "Bâtiment"
and finds it by typing "immeuble", while the box still receives "building".

This module is the offline fallback catalogue. When the server catalogue is
reachable (see ``segmentation_presets_client``) it is merged on top, but the
shapes are identical so the gallery renders either source the same way. The
object set + strong/weak flags are aerial-imagery object classes phrased under
the model's "short noun phrase" rule.

Two sets of strings here are frozen and must never be reworded: every ``prompt``
(the billed, routed token) and every preset ``id`` (keys the demo images and the
local favorites store). Category ``key`` values are frozen too: the server
``detection_policy`` maps them to shape classes.
"""

from __future__ import annotations

from qgis.PyQt.QtCore import QSettings

from .segmentation_search_terms import preset_search_terms

LANGS = ("en", "fr", "es", "pt", "de", "it", "nl", "pl", "id", "ja", "zh_CN", "zh_TW")

# The "Popular" tab, in display order: the objects users ask for most. Also the
# single source of each preset's ``top_pick`` flag, so the tab and the flag can
# never disagree.
#
# Popular is the first thing a new user sees, so an entry here must have a demo
# image. Not every object has one yet, and a blank card in this tab costs more
# than picking the next object down the demand list. Anything added here needs
# its before/after seeded first.
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
    """Build one preset. ``prompt`` is the English cloud-model token (lowercase)."""
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


# Extra-locale label overlays (de/it/nl/pl/id/ja/zh_CN/zh_TW), merged on top of the
# inline en/fr/es/pt labels by ``_p``/``_cat``. The cloud-model ``prompt`` token stays
# English in every locale; only the browsed label follows the UI language. A missing
# entry falls back to English via ``pick_label``, so this table need not be exhaustive.
_L10N_LANGS = ("de", "it", "nl", "pl", "id", "ja", "zh_CN", "zh_TW")


def _l(de: str, it: str, nl: str, pl: str, id_: str, ja: str, zh_cn: str, zh_tw: str) -> dict:
    return dict(zip(_L10N_LANGS, (de, it, nl, pl, id_, ja, zh_cn, zh_tw)))


# Category labels spell out "and" instead of "&": Qt eats a single ampersand in
# a button label as the mnemonic marker, and the sidebar entries are buttons.
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

# Keyed lookup, so the order of this table is free and does not track the
# catalogue order below.
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
    "train": _l("Zug", "Treno", "Trein", "Pociąg", "Kereta", "列車", "火车", "火車"),
    "parking_lot": _l(
        "Parkplatz", "Parcheggio", "Parkeerplaats", "Parking", "Tempat parkir", "駐車場", "停车场", "停車場"
    ),
    "bridge": _l("Brücke", "Ponte", "Brug", "Most", "Jembatan", "橋", "桥梁", "橋"),
    "roundabout": _l("Kreisverkehr", "Rotatoria", "Rotonde", "Rondo", "Bundaran", "ロータリー", "环岛", "圓環"),
    "runway": _l("Landebahn", "Pista", "Startbaan", "Pas startowy", "Landasan pacu", "滑走路", "跑道", "跑道"),
    "sidewalk": _l("Gehweg", "Marciapiede", "Stoep", "Chodnik", "Trotoar", "歩道", "人行道", "人行道"),
    "airplane": _l("Flugzeug", "Aereo", "Vliegtuig", "Samolot", "Pesawat", "飛行機", "飞机", "飛機"),
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
    "tree_canopy": _l(
        "Baumkronen", "Chioma arborea", "Boomkruinen", "Korony drzew", "Tajuk pohon", "樹冠", "树冠", "樹冠"
    ),
    "bush": _l("Strauch", "Cespuglio", "Struik", "Krzak", "Semak", "灌木", "灌木", "灌木"),
    "vineyard": _l("Weinberg", "Vigneto", "Wijngaard", "Winnica", "Kebun anggur", "ぶどう畑", "葡萄园", "葡萄園"),
    "farm_field": _l("Acker", "Campo agricolo", "Akker", "Pole uprawne", "Lahan pertanian", "農地", "农田", "農田"),
    "lawn": _l("Rasenfläche", "Prato", "Gazon", "Trawnik", "Halaman berumput", "芝地", "草坪", "草坪"),
    "hedge": _l("Hecke", "Siepe", "Haag", "Żywopłot", "Pagar tanaman", "生垣", "绿篱", "綠籬"),
    "orchard": _l("Obstanlage", "Frutteto", "Boomgaard", "Sad", "Kebun buah", "果樹園", "果园", "果園"),
    "crop_field": _l(
        "Kulturfläche",
        "Campo coltivato",
        "Landbouwperceel",
        "Uprawa rolna",
        "Ladang tanaman",
        "耕作地",
        "作物田",
        "作物田",
    ),
    "soil": _l("Bodenfläche", "Terreno", "Bodem", "Gleba", "Tanah", "土壌", "土壤", "土壤"),
    "stone": _l("Stein", "Pietra", "Steen", "Kamień", "Batu", "岩石", "石块", "石塊"),
    "pond": _l("Teich", "Stagno", "Vijver", "Staw", "Kolam", "池", "池塘", "池塘"),
    "river": _l("Fluss", "Fiume", "Rivier", "Rzeka", "Sungai", "河川", "河流", "河流"),
    "roof": _l(
        "Dachfläche",
        "Copertura edilizia",
        "Dakvlak",
        "Pokrycie dachowe",
        "Atap bangunan",
        "建物屋根",
        "屋面",
        "建築屋面",
    ),
    "driveway": _l("Zufahrt", "Accesso carrabile", "Oprit", "Podjazd", "Jalan masuk", "進入路", "车道", "車道"),
    "taxiway": _l(
        "Rollbahn",
        "Via di rullaggio",
        "Taxibaan",
        "Droga kołowania",
        "Jalur taxi",
        "誘導路",
        "滑行道",
        "滑行道",
    ),
    "water_tower": _l(
        "Wasserturm",
        "Torre piezometrica",
        "Watertoren",
        "Wieża ciśnień",
        "Menara air",
        "配水塔",
        "水塔",
        "水塔",
    ),
}


# Sidebar emoji by category key. Covers BOTH the offline taxonomy keys below
# and the richer server taxonomy (different keys) so the gallery sidebar always
# shows an icon, even when the server catalogue omits the emoji field (it
# currently sends emoji: null). Keep the server keys in sync with
# /api/ai-segmentation/presets.
_CATEGORY_EMOJI: dict[str, str] = {
    # offline taxonomy
    "buildings": "\U0001f3e2",  # building
    "vegetation": "\U0001f333",  # deciduous tree
    "transport": "\U0001f6e3️",  # motorway
    "land_water": "\U0001f4a7",  # droplet
    "vehicles_transport": "\U0001f697",  # car
    "agriculture": "\U0001f33e",  # sheaf of rice
    "energy": "⚡",  # high voltage
    "sport_recreation": "\U0001f3df️",  # stadium
    "aircraft_vessels": "✈️",  # airplane
    "industry": "\U0001f3ed",  # factory
    # server taxonomy, plus keys this catalogue used before
    "energy_industrial": "⚡",  # high voltage
    "vehicles": "\U0001f697",  # car
    "aircraft_maritime": "✈️",  # airplane
    "water": "\U0001f4a7",  # droplet
    "sports": "\U0001f3df️",  # stadium
    "land": "\U0001f3d4️",  # snow-capped mountain
}
_CATEGORY_EMOJI_DEFAULT = "\U0001f4c2"  # open folder: generic category icon


def category_emoji(key: str) -> str:
    """Sidebar emoji for a category key, falling back to a generic folder glyph
    so an unknown/new server key still renders an icon."""
    return _CATEGORY_EMOJI.get(key, _CATEGORY_EMOJI_DEFAULT)


# Catalogue grouped by OBJECT FAMILY (what the object is), the convention used
# by aerial-imagery object datasets. Discrete countable objects are browsed by
# family, not by GIS use-domain (that taxonomy fits continuous land cover, not
# objects). Categories run most-asked-for first, and so do the presets inside
# each one. An object earns a card by being asked for, so this list is curated
# rather than exhaustive; the few weak/continuous classes carry weak=True so
# the UI can flag them.
_CATEGORIES: list[dict] = [
    _cat(
        "buildings",
        "Buildings and rooftops",
        "Bâtiments et toitures",
        "Edificios y tejados",
        "Edifícios e telhados",
        [
            _p("building", "building", "Building", "Bâtiment", "Edificio", "Edifício"),
            _p("rooftop", "rooftop", "Rooftop", "Toiture", "Tejado", "Telhado"),
            _p("house", "house", "House", "Maison", "Casa", "Casa"),
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
            _p("roof", "roof", "Roof", "Toit", "Cubierta", "Cobertura"),
            _p("water_tower", "water tower", "Water tower", "Château d'eau", "Torre de agua", "Torre de água"),
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
            _p("tree_canopy", "tree canopy", "Tree canopy", "Canopée", "Dosel arbóreo", "Dossel arbóreo", weak=True),
            _p("bush", "bush", "Bush", "Buisson", "Arbusto", "Arbusto"),
            _p("grass", "grass", "Grass", "Herbe", "Hierba", "Grama", weak=True),
            _p("vegetation", "vegetation", "Vegetation", "Végétation", "Vegetación", "Vegetação", weak=True),
            _p("lawn", "lawn", "Lawn", "Pelouse", "Césped", "Gramado", weak=True),
            _p("hedge", "hedge", "Hedge", "Haie", "Seto", "Sebe"),
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
            _p("sidewalk", "sidewalk", "Sidewalk", "Trottoir", "Acera", "Calçada"),
            _p("bridge", "bridge", "Bridge", "Pont", "Puente", "Ponte"),
            _p("roundabout", "roundabout", "Roundabout", "Rond-point", "Rotonda", "Rotatória"),
            _p("runway", "runway", "Runway", "Piste", "Pista", "Pista"),
            _p("dock", "dock", "Dock", "Quai", "Muelle", "Doca"),
            _p("driveway", "driveway", "Driveway", "Allée carrossable", "Acceso vehicular", "Acesso de garagem"),
            _p("taxiway", "taxiway", "Taxiway", "Voie de circulation", "Calle de rodaje", "Pista de táxi"),
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
            _p("bare_ground", "bare ground", "Bare ground", "Sol nu", "Suelo desnudo", "Solo exposto", weak=True),
            _p("rock", "rock", "Rock", "Roche", "Roca", "Rocha"),
            _p("quarry", "quarry", "Quarry", "Carrière", "Cantera", "Pedreira"),
            _p("soil", "soil", "Soil", "Terrain nu", "Tierra", "Solo", weak=True),
            _p("stone", "stone", "Stone", "Pierre", "Piedra", "Pedra"),
            _p("pond", "pond", "Pond", "Mare", "Estanque", "Lagoa"),
            _p("river", "river", "River", "Rivière", "Río", "Rio", weak=True),
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
            _p("truck", "truck", "Truck", "Camion", "Camión", "Caminhão"),
            _p("train", "train", "Train", "Train", "Tren", "Trem"),
        ],
    ),
    _cat(
        "agriculture",
        "Fields and crops",
        "Parcelles et cultures",
        "Parcelas y cultivos",
        "Talhões e culturas",
        [
            # No weak flag on the two parcel presets. `weak` does two unrelated
            # jobs: it shows a "fuzzy edges" note in the library, and it routes
            # the prompt to MAP merging in _default_merge_separate. A parcel
            # has its own boundary and is counted one by one, so the MAP route
            # is wrong here, and it is what shipped a whole farm as a single
            # object. Losing the note on these two costs a hint line; keeping
            # the flag cost the layer.
            _p(
                "farm_field",
                "farm field",
                "Farm field",
                "Parcelle agricole",
                "Parcela agrícola",
                "Talhão agrícola",
            ),
            _p("field", "field", "Field", "Champ", "Campo", "Campo"),
            _p("vineyard", "vineyard", "Vineyard", "Vigne", "Viñedo", "Vinhedo"),
            _p("orchard", "orchard", "Orchard", "Verger", "Huerto frutal", "Pomar"),
            _p("crop_field", "crop field", "Crop field", "Culture", "Terreno de cultivo", "Área de cultivo"),
        ],
    ),
    _cat(
        "energy",
        "Solar and wind energy",
        "Énergie solaire et éolienne",
        "Energía solar y eólica",
        "Energia solar e eólica",
        [
            _p("solar_panel", "solar panel", "Solar panel", "Panneau solaire", "Panel solar", "Painel solar"),
            _p("wind_turbine", "wind turbine", "Wind turbine", "Éolienne", "Aerogenerador", "Turbina eólica"),
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
            _p("tennis_court", "tennis court", "Tennis court", "Court de tennis", "Pista de tenis", "Quadra de tênis"),
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
        "aircraft_vessels",
        "Aircraft and boats",
        "Avions et bateaux",
        "Aviones y barcos",
        "Aviões e barcos",
        [
            _p("airplane", "airplane", "Airplane", "Avion", "Avión", "Avião"),
            _p("boat", "boat", "Boat", "Bateau", "Bote", "Barco"),
        ],
    ),
    _cat(
        "industry",
        "Industry and works",
        "Industrie et chantiers",
        "Industria y obras",
        "Indústria e obras",
        [
            _p("construction_site", "construction site", "Construction site", "Chantier", "Obra", "Canteiro de obras"),
            _p("crane", "crane", "Crane", "Grue", "Grúa", "Guindaste"),
            _p(
                "shipping_container", "shipping container", "Shipping container", "Conteneur", "Contenedor", "Contêiner"
            ),
        ],
    ),
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


def fallback_categories() -> list[dict]:
    """The offline catalogue (ordered domains, each with its presets)."""
    return _CATEGORIES


def catalog_revision() -> str:
    """A stamp that changes whenever the cached served catalogue changes.

    Anything that builds an index over the catalogue keys it on this, so a
    background refresh landing while a widget is open is picked up on the next
    read instead of being missed for the rest of the session. Empty when
    nothing was ever fetched, which is the shipped-fallback case.
    """
    try:
        # Lazy import: the client module imports this one for its fallbacks.
        # Its cache timestamp is the one value that moves on every refresh, so
        # it is read from there rather than spelled out a second time here.
        from .segmentation_presets_client import _CACHE_TS_KEY

        return str(QSettings().value(_CACHE_TS_KEY, "") or "")
    except Exception:  # noqa: BLE001 -- no cache is the normal offline case
        return ""


def merged_categories(served: list[dict] | None) -> list[dict]:
    """The served categories with anything the shipped catalogue has and they
    do not appended, keyed by prompt token.

    A served catalogue is authoritative for every object it carries: its label,
    its search words and its order all win, and nothing here rewrites them. But
    a cache is kept and reused for as long as the fetch keeps failing, so an
    object shipped in a later plugin version stayed invisible behind a cache
    written before it existed. Appending the missing ones costs the served
    catalogue nothing and gives the new objects somewhere to appear.

    Every shape is checked on the way in, because this reads JSON off the
    network and off a cache written by an older plugin: a category that is not
    a dict, a ``presets`` value that is not a list, and a preset that is not a
    dict are all dropped. Iterating one of them instead raises on a read path
    the whole library and the prompt box sit on, which turns one bad payload
    into a plugin that cannot list a single object.
    """
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
        # A copy, never the served dict: the list below is rebuilt on it, and
        # the caller may be holding the object the fetch handed back.
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
        # A rebuilt list, never an append: the served list object may be held
        # by the caller and would otherwise grow on every read.
        target["presets"] = list(target.get("presets") or []) + missing
    return out


def all_presets() -> list[dict]:
    return [p for cat in _CATEGORIES for p in cat["presets"]]


def known_tokens() -> list[str]:
    """Flat, de-duplicated English prompt tokens (for the validator's
    'did you mean' suggestions and its typo repair).

    Read from the LIVE catalogue, the cached server one when there is one and
    the shipped fallback otherwise, for the same reason
    ``token_by_localized_label`` does: these tokens are the whole pool the typo
    corrector may repair a word into, so an object added on the server was
    invisible to it until the next plugin release. A user typing a near miss of
    a brand new object got no repair and no suggestion.
    """
    cats: list[dict] = []
    try:
        # Lazy import: the client module imports this one for its fallbacks.
        from .segmentation_presets_client import cached_or_offline_catalog

        cats, _top = cached_or_offline_catalog()
    except Exception:  # noqa: BLE001 -- offline fallback below
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


# Letters no normal form decomposes into a base plus an accent, spelled out by
# hand so a Polish or Nordic label still answers a query typed without them.
_LETTERS_THAT_DO_NOT_DECOMPOSE = str.maketrans(
    {"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D",
     "ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE"}
)


def fold_search_text(text) -> str:
    """Fold a label or a query to lowercase without accents ('Bâtiment' ->
    'batiment'), so a user who skips the accents still finds the object.

    Only the accents go. Everything else is kept, Japanese and Chinese
    included: dropping to ASCII emptied both the label and the query in those
    scripts, which made every search there match nothing."""
    import unicodedata

    folded = str(text or "").translate(_LETTERS_THAT_DO_NOT_DECOMPOSE)
    decomposed = unicodedata.normalize("NFKD", folded)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower().strip()


def search_terms_of(preset) -> list[str]:
    """Every extra search word a preset carries, in every language it carries.

    Reads the OPTIONAL ``search_terms`` field, which the server catalogue may
    omit or send in another shape (a dict of strings, a flat list, or one
    string). Anything unreadable yields an empty list, never an exception.
    """
    if not isinstance(preset, dict):
        return []
    terms = preset.get("search_terms")
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
    """One accent-folded string to substring-match a search query against: the
    prompt token, the label in EVERY shipped language, the category, and the
    extra terms.

    Every language, not just the interface one: people mix languages in a
    search box, and a French user on an English QGIS still types "eolienne".
    Same reasoning as the prompt box's silent translation.
    """
    labels = (preset or {}).get("label")
    parts = [str((preset or {}).get("prompt", "")), category_label]
    if isinstance(labels, dict):
        parts.extend(str(v) for v in labels.values())
    else:
        parts.append(pick_label(labels, ""))
    parts.extend(search_terms_of(preset))
    return fold_search_text(" ".join(p for p in parts if p))


def preset_matches_query(preset, query: str, category_label: str = "") -> bool:
    """Does this preset answer what the user typed? Folded substring match, so
    'batiment', 'Bâtiment' and 'immeuble' all reach the building card."""
    folded = fold_search_text(query)
    return bool(folded) and folded in preset_search_haystack(preset, category_label)


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
