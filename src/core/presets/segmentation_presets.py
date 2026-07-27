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
    "lake",
    "river",
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
        "search_terms": _preset_search_terms(pid),
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
    "railway_track": _l(
        "Eisenbahnstrecke", "Binario", "Spoorweg", "Tor kolejowy", "Rel kereta", "線路", "铁轨", "鐵軌"
    ),
    "airplane": _l("Flugzeug", "Aereo", "Vliegtuig", "Samolot", "Pesawat", "飛行機", "飞机", "飛機"),
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
    "lake": _l("See", "Lago", "Meer", "Jezioro", "Danau", "湖", "湖泊", "湖泊"),
    "river": _l("Fluss", "Fiume", "Rivier", "Rzeka", "Sungai", "河川", "河流", "河川"),
    "dam": _l("Damm", "Diga", "Dam", "Zapora", "Bendungan", "ダム", "大坝", "水壩"),
}

# Extra words the search box should also match, per preset id, as
# ``(en, fr, es, pt)`` space-separated lists. They cover what a user types when
# it is not the label: a synonym, a plural, a regional word, or the same word
# without its accents (the search compares raw text, so both spellings earn
# their place). Optional by design: a preset with no entry gets an empty dict,
# and a catalogue payload without the field still searches on label + prompt.
_SEARCH_TERMS: dict[str, tuple[str, str, str, str]] = {
    "building": ("structure footprint block outline", "immeuble construction emprise batiment bati",
                 "inmueble construccion edificacion huella", "imovel construcao edificacao"),
    "rooftop": ("roof roofs tiles roof outline", "toit toits toiture couverture",
                "techo techos cubierta azotea", "telhado telhados cobertura laje"),
    "house": ("home dwelling residential villa", "maison pavillon habitation villa",
              "vivienda chalet casas residencial", "moradia residencia casas"),
    "warehouse": ("depot logistics hangar industrial building", "entrepot hangar logistique depot",
                  "nave industrial bodega deposito almacen", "armazem galpao deposito logistica"),
    "greenhouse": ("glasshouse polytunnel nursery", "serre serres tunnel horticulture",
                   "invernaderos umbraculo vivero", "estufa estufas viveiro"),
    "shed": ("outbuilding hut barn garage", "abri remise grange garage",
             "caseta cobertizo granero garaje", "barracao abrigo celeiro garagem"),
    "silo": ("grain silo granary farm silo", "silos grenier", "silos granero", "silos celeiro"),
    "storage_tank": ("tank oil tank fuel tank cistern", "cuve citerne reservoir",
                     "tanque cisterna deposito", "tanque cisterna reservatorio"),
    "tree": ("trees canopy crown forest woodland", "arbres houppier canopee foret forêt",
             "arboles copa dosel bosque", "arvores copa dossel floresta"),
    "forest": ("woodland woods forestry timber tree stand canopy", "foret bois boisement massif forestier canopee",
               "bosques arbolado selva masa forestal dosel", "florestas mata bosque arvoredo dossel"),
    "hedge": ("hedgerow windbreak shrub row", "haies bocage brise-vent",
              "setos seto vivo", "sebe cerca viva"),
    "bush": ("shrub scrub brush bushes", "arbuste broussaille fourre",
             "matorral arbustos maleza", "arbustos moita mato"),
    "grass": ("lawn turf meadow pasture grassland green space", "pelouse gazon prairie paturage espace vert",
              "cesped pasto pradera prado zona verde", "gramado relva pasto pastagem area verde"),
    "vegetation": ("vegetated green cover greenery plants undergrowth scrub",
                   "vegetal couvert vegetal verdure plantes broussaille",
                   "cubierta vegetal verde plantas maleza", "cobertura vegetal verde plantas mato"),
    "road": ("street highway motorway asphalt pavement", "rue chaussee autoroute voirie bitume",
             "calle via autopista pavimento asfalto", "rua via rodovia asfalto pavimento"),
    "parking_lot": ("car park parking parking space", "stationnement aire de stationnement places",
                    "aparcamiento playa de estacionamiento", "estacionamento vaga patio"),
    "bridge": ("viaduct overpass flyover footbridge", "viaduc passerelle ouvrage d'art",
               "viaducto paso elevado pasarela", "viaduto passarela"),
    "roundabout": ("traffic circle rotary junction", "giratoire carrefour rond point",
                   "glorieta redondel", "rotula retorno"),
    "railway_track": ("railway railroad rail tracks train line", "rail chemin de fer voies",
                      "ferrocarril vias del tren riel", "trilhos linha ferrea ferroviaria"),
    "runway": ("airstrip taxiway apron airport", "aerodrome aeroport taxiway",
               "pista de aterrizaje aeropuerto", "pista de pouso aeroporto"),
    "dock": ("quay wharf pier jetty marina port harbour harbor", "quai ponton jetee port marina",
             "muelle embarcadero dique puerto", "cais pier atracadouro porto"),
    "water": ("water body waterbody surface water pond basin wetland flood",
              "eau plan d'eau etendue d'eau mare bassin zone humide",
              "agua cuerpo de agua masa de agua estanque humedal",
              "agua corpo dagua massa de agua lagoa acude"),
    "lake": ("pond reservoir water body water", "etang plan d'eau reservoir eau",
             "laguna embalse agua", "lagoa represa agua"),
    "river": ("stream creek canal watercourse water", "fleuve ruisseau cours d'eau canal",
              "arroyo cauce canal", "riacho corrego canal"),
    "dam": ("weir reservoir hydro dyke", "digue retenue ecluse",
            "represa dique embalse", "represa dique acude"),
    "bare_ground": ("bare soil dirt sand cleared land earth", "terre nue sable terrain nu remblai",
                    "tierra desnuda arena terreno despejado", "solo nu areia terreno limpo"),
    "rock": ("rocks stone boulder outcrop bedrock rocky", "roches pierre rocher affleurement rocheux",
             "rocas piedra canto rodado afloramiento rocoso", "rochas pedra pedregulho afloramento rochoso"),
    "quarry": ("mine pit gravel pit open pit extraction", "mine graviere carriere extraction",
               "mina gravera cantera", "mina cascalheira mineracao"),
    "car": ("vehicle auto automobile cars van", "vehicule voitures automobile camionnette",
            "vehiculo automovil autos furgoneta", "veiculo automovel carros van"),
    "truck": ("lorry semi hgv freight van", "poids lourd semi-remorque fourgon",
              "camiones trailer furgon", "caminhoes carreta furgao"),
    "train": ("railcar wagon locomotive rolling stock", "wagon locomotive rame",
              "vagon locomotora tren", "vagao locomotiva composicao"),
    "farm_field": ("field cropland farmland parcel crop agriculture", "champ parcelle culture terre agricole",
                   "campo cultivo parcela agricultura", "campo lavoura cultura agricultura"),
    "field": ("fields cropland farmland parcel plot paddock crop", "champs parcelle culture terrain lopin",
              "campos parcela cultivo terreno lote", "campos talhao parcela cultivo terreno lote"),
    "vineyard": ("vines grapes wine", "vignoble vignes parcelle viticole",
                 "vina vinedo parras", "vinha parreiral uva"),
    "orchard": ("fruit trees plantation grove olive grove", "plantation oliveraie arbres fruitiers",
                "plantacion frutales olivar", "plantacao olival frutas"),
    "solar_panel": ("pv photovoltaic solar module rooftop solar", "photovoltaique pv module solaire",
                    "fotovoltaico pv placa solar modulo", "fotovoltaico pv placa solar modulo"),
    "solar_farm": ("pv plant solar park photovoltaic plant solar array", "centrale solaire parc photovoltaique",
                   "parque solar planta fotovoltaica huerto solar", "usina fotovoltaica parque solar"),
    "wind_turbine": ("windmill wind farm turbine wind power", "aerogenerateur parc eolien moulin",
                     "eolico molino parque eolico", "eolica cata-vento parque eolico"),
    "swimming_pool": ("pool pools swimming water", "piscines bassin eau",
                      "alberca pileta piscinas", "piscinas tanque"),
    "tennis_court": ("tennis court padel", "tennis terrain padel", "tenis cancha padel", "tenis quadra padel"),
    "soccer_field": ("football pitch sports field", "football terrain de sport foot",
                     "futbol cancha campo deportivo", "futebol campo quadra"),
    "running_track": ("athletics track oval", "athletisme piste stade",
                      "atletismo pista tartan", "atletismo pista tartan"),
    "stadium": ("arena sports ground grandstand", "arene enceinte sportive tribune",
                "arena gradas coliseo", "arena arquibancada"),
    "airplane": ("plane aircraft jet aviation", "avions aeronef jet aviation",
                 "aviones aeronave jet", "avioes aeronave jato"),
    "ship": ("vessel cargo ship tanker freighter", "cargo petrolier navires bateau",
             "buque carguero nave", "navios cargueiro embarcacao"),
    "boat": ("vessel yacht dinghy sailboat canoe", "barque voilier yacht embarcation",
             "barca velero lancha yate", "lancha veleiro canoa"),
    "construction_site": ("building site works excavation earthworks", "chantiers travaux terrassement",
                          "obras construccion movimiento de tierras", "obras canteiro terraplenagem"),
    "crane": ("tower crane gantry port crane", "grues portique", "gruas portico", "guindastes portico"),
    "shipping_container": ("container containers freight intermodal", "conteneurs container caisse",
                           "contenedores contenedor maritimo", "conteineres container"),
    "chimney": ("smokestack stack flue cooling tower", "cheminees fumee tour de refroidissement",
                "chimeneas torre de enfriamiento", "chamines torre de resfriamento"),
}

_SEARCH_TERM_LANGS = ("en", "fr", "es", "pt")


def _preset_search_terms(pid: str) -> dict[str, str]:
    """The extra search words for one preset, empty when it has none."""
    row = _SEARCH_TERMS.get(pid)
    return dict(zip(_SEARCH_TERM_LANGS, row)) if row else {}


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
            _p("forest", "forest", "Forest", "Forêt", "Bosque", "Floresta", weak=True),
            _p("hedge", "hedge", "Hedge", "Haie", "Seto", "Cerca viva"),
            _p("bush", "bush", "Bush", "Buisson", "Arbusto", "Arbusto"),
            _p("grass", "grass", "Grass", "Herbe", "Hierba", "Grama", weak=True),
            _p("vegetation", "vegetation", "Vegetation", "Végétation", "Vegetación", "Vegetação", weak=True),
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
            _p("bridge", "bridge", "Bridge", "Pont", "Puente", "Ponte"),
            _p("roundabout", "roundabout", "Roundabout", "Rond-point", "Rotonda", "Rotatória"),
            _p("railway_track", "railway track", "Railway track", "Voie ferrée", "Vía férrea", "Ferrovia"),
            _p("runway", "runway", "Runway", "Piste", "Pista", "Pista"),
            _p("dock", "dock", "Dock", "Quai", "Muelle", "Doca"),
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
            _p("lake", "lake", "Lake", "Lac", "Lago", "Lago"),
            _p("river", "river", "River", "Rivière", "Río", "Rio", weak=True),
            _p("dam", "dam", "Dam", "Barrage", "Presa", "Barragem"),
            _p("bare_ground", "bare ground", "Bare ground", "Sol nu", "Suelo desnudo", "Solo exposto", weak=True),
            _p("rock", "rock", "Rock", "Roche", "Roca", "Rocha"),
            _p("quarry", "quarry", "Quarry", "Carrière", "Cantera", "Pedreira"),
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
            _p(
                "farm_field",
                "farm field",
                "Farm field",
                "Parcelle agricole",
                "Parcela agrícola",
                "Talhão agrícola",
                weak=True,
            ),
            _p("field", "field", "Field", "Champ", "Campo", "Campo", weak=True),
            _p("vineyard", "vineyard", "Vineyard", "Vigne", "Viñedo", "Vinhedo"),
            _p("orchard", "orchard", "Orchard", "Verger", "Huerto", "Pomar"),
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
            _p("solar_farm", "solar farm", "Solar farm", "Ferme solaire", "Planta solar", "Usina solar"),
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
            _p("ship", "ship", "Ship", "Navire", "Barco", "Navio"),
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
            _p("chimney", "chimney", "Chimney", "Cheminée", "Chimenea", "Chaminé"),
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


def all_presets() -> list[dict]:
    return [p for cat in _CATEGORIES for p in cat["presets"]]


def known_tokens() -> list[str]:
    """Flat, de-duplicated English prompt tokens (for the validator's
    'did you mean' suggestions)."""
    seen: dict[str, None] = {}
    for p in all_presets():
        seen.setdefault(p["prompt"], None)
    return list(seen.keys())


def fold_search_text(text) -> str:
    """Accent-fold a label or a query to lowercase ASCII ('Bâtiment' ->
    'batiment'), so a user who skips the accents still finds the object."""
    import unicodedata

    return unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii").lower().strip()


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
    for cat in cats or _CATEGORIES:
        for p in cat.get("presets", []):
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
