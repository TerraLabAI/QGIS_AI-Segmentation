"""Search synonyms for the segment library, per preset id and per language.

These are the EXTRA words the search box matches on top of a preset's label and
its cloud-model prompt token: a synonym, a plural, a regional word, or the same
word without its accents (the search compares folded text, so both spellings
earn their place). They are not translations of the label, which is indexed
already: they are what a user types when the label is not the word they know.

The table covers the twelve catalogue languages, for the Popular presets only:
the full table travels with the served catalogue, which overrides this one and
must keep the same shape. Optional by design: a preset with no row gets an
empty dict, and a catalogue payload without the field still searches on
label + prompt.
"""

from __future__ import annotations

_SEARCH_TERMS: dict[str, dict[str, str]] = {
    "building": {"en": "structure footprint block outline",
                 "fr": "immeuble construction emprise batiment bati",
                 "es": "inmueble construccion edificacion huella",
                 "pt": "imovel construcao edificacao",
                 "de": "bauwerk gebäude grundriss gebäudeumriss block",
                 "it": "edificio struttura impronta sagoma",
                 "nl": "gebouw bebouwing pand omtrek contour",
                 "pl": "budynek obrys budynku zabudowa konstrukcja",
                 "id": "bangunan struktur tapak gedung konstruksi",
                 "ja": "建物 建築物 建物輪郭 建物フットプリント",
                 "zh_CN": "建筑物 建筑 轮廓 建筑占地",
                 "zh_TW": "建築物 建物 建築輪廓 建物外框 建築占地"},
    "house": {"en": "home dwelling residential villa",
              "fr": "maison pavillon habitation villa",
              "es": "vivienda chalet casas residencial",
              "pt": "moradia residencia casas",
              "de": "haus wohnhaus wohngebäude villa",
              "it": "casa abitazione residenza villa",
              "nl": "huis woning woonhuis villa",
              "pl": "dom budynek mieszkalny willa siedlisko",
              "id": "rumah hunian tempat tinggal vila",
              "ja": "家 住宅 一戸建て 民家 邸宅",
              "zh_CN": "房屋 住宅 民居 别墅",
              "zh_TW": "房屋 住宅 民宅 住家 別墅"},
    "tree": {"en": "trees canopy crown forest woodland fruit trees plantation grove tree rows apple orchard",
             "fr": "arbres houppier canopee foret forêt verger arbres fruitiers plantation bosquet rangées arbres "
                   "pommeraie orchard",
             "es": "arboles copa dosel bosque frutales plantación huerto arboleda manzanos orchard huerto frutal",
             "pt": "arvores copa dossel floresta pomar árvores frutíferas plantação bosque fileiras árvores orchard",
             "de": "bäume baumkrone kronendach wald gehölz obstgarten obstplantage baumgruppe baumreihen apfelplantage "
                   "orchard obstanlage",
             "it": "alberi chioma corona bosco foresta frutteto alberi da frutto piantagione filare di alberi meleti "
                   "filari orchard",
             "nl": "boom bomen kruin kroon bos bosgebied boomgaard fruitbomen plantage boomgroep bomenrijen "
                   "appelboomgaard orchard",
             "pl": "drzewo drzewa korona drzew las zagajnik sad drzewa owocowe plantacja gaj rzędy drzew sad "
                   "jabłoniowy orchard",
             "id": "pohon pepohonan tajuk mahkota hutan kebun buah perkebunan pohon buah rumpun barisan pohon orchard",
             "ja": "木 樹木 樹冠 林 森林 果樹園 果樹農園 樹木園 果樹列 りんご園 orchard",
             "zh_CN": "树木 树冠 林地 森林 果园 果树园 植物园 林地 果树行 苹果园 orchard",
             "zh_TW": "樹木 樹冠 林木 森林 林地 果園 果樹園 果樹種植地 林園 果樹列 蘋果園 orchard"},
    "road": {"en": "street highway motorway asphalt pavement traffic circle rotary junction roundabout",
             "fr": "rue chaussee autoroute voirie bitume giratoire carrefour rond point roundabout rond-point",
             "es": "calle via autopista pavimento asfalto glorieta redondel roundabout rotonda",
             "pt": "rua via rodovia asfalto pavimento rotula retorno roundabout rotatória",
             "de": "straße fahrbahn landstraße autobahn asphalt verkehrsweg kreisverkehr verkehrskreisel kreisförmige "
                   "kreuzung roundabout",
             "it": "strada via autostrada superstrada asfalto carreggiata rotonda rotatoria rondò incrocio circolare "
                   "roundabout",
             "nl": "weg straat snelweg autoweg asfalt rijbaan rotonde verkeersplein verkeerscirkel kruispunt "
                   "roundabout",
             "pl": "droga ulica szosa autostrada nawierzchnia asfaltowa jezdnia rondo skrzyżowanie okrężne rondo "
                   "drogowe wyspa centralna roundabout",
             "id": "jalan ruas jalan jalan raya aspal perkerasan bundaran lingkaran lalu lintas simpang melingkar "
                   "roundabout",
             "ja": "道路 通り 幹線道路 高速道路 舗装路 車道 ラウンドアバウト ロータリー 環状交差点 円形交差点 "
                   "roundabout",
             "zh_CN": "道路 街道 公路 高速公路 沥青路 路面 环岛 环形交叉口 转盘式路口 roundabout",
             "zh_TW": "道路 街道 公路 高速公路 柏油路 路面 圓環 環島 迴轉道 環形路口 roundabout"},
    "parking_lot": {"en": "car park parking parking space",
                    "fr": "stationnement aire de stationnement places",
                    "es": "aparcamiento playa de estacionamiento",
                    "pt": "estacionamento vaga patio",
                    "de": "parkplatz stellplätze parkfläche parken autoabstellplatz",
                    "it": "parcheggio area di sosta stalli auto",
                    "nl": "parkeerplaats parkeerterrein parking parkeervak autoparkeerplaats",
                    "pl": "parking plac parkingowy miejsce parkingowe parking samochodowy",
                    "id": "parkiran tempat parkir lahan parkir area parkir",
                    "ja": "駐車場 パーキング 駐車スペース 車置き場",
                    "zh_CN": "停车场 停车区 停车位 车辆停放区",
                    "zh_TW": "停車場 停車區 停車位 汽車停放區"},
    "water": {"en": "water body waterbody surface water pond basin wetland flood lake river stream reservoir stream "
                    "watercourse creek channel waterway",
              "fr": "eau plan d'eau etendue d'eau mare bassin zone humide lac riviere etang rivière fleuve cours d'eau "
                    "ruisseau canal chenal river",
              "es": "agua cuerpo de agua masa de agua estanque humedal lago rio río arroyo riachuelo cauce canal "
                    "corriente river",
              "pt": "agua corpo dagua massa de agua lagoa acude lago rio rio córrego riacho canal curso agua arroio "
                    "river",
              "de": "gewässer wasserfläche teich becken feuchtgebiet see fluss bach reservoir bach fließgewässer "
                    "wasserlauf kanal wasserweg flussbett river",
              "it": "acqua specchio d acqua bacino stagno lago fiume torrente fiume torrente corso d acqua canale "
                    "alveo via d acqua river",
              "nl": "water wateroppervlak waterlichaam vijver bekken moeras meer rivier beek reservoir rivier beek "
                    "waterloop kreek kanaal watergang river",
              "pl": "woda zbiornik wodny akwen staw jezioro rzeka potok mokradło rzeka strumień ciek wodny potok kanał "
                    "koryto wodne droga wodna river",
              "id": "badan air permukaan air kolam danau sungai waduk rawa sungai anak sungai aliran air kanal saluran "
                    "air river",
              "ja": "水域 水面 池 湿地 湖 河川 貯水池 川 河川 小川 水路 渓流 河道 river",
              "zh_CN": "水体 水面 湖泊 河流 水塘 湿地 水库 河流 溪流 河道 水道 沟渠 水路 river",
              "zh_TW": "水域 水體 水面 池塘 湖泊 河流 濕地 溪流 河道 小溪 水道 河渠 水路 river"},
    "car": {"en": "vehicle auto automobile cars van",
            "fr": "vehicule voitures automobile camionnette",
            "es": "vehiculo automovil autos furgoneta",
            "pt": "veiculo automovel carros van",
            "de": "auto fahrzeug pkw wagen transporter",
            "it": "auto automobile vettura veicolo macchina",
            "nl": "auto voertuig personenauto wagen bestelbus",
            "pl": "samochód auto pojazd samochody furgonetka",
            "id": "mobil kendaraan otomobil van",
            "ja": "車 自動車 乗用車 車両 バン",
            "zh_CN": "汽车 轿车 小汽车 车辆 面包车",
            "zh_TW": "汽車 車輛 小客車 轎車 廂型車"},
    "solar_panel": {"en": "pv photovoltaic solar module rooftop solar solar farm solar park",
                    "fr": "photovoltaique pv module solaire centrale solaire",
                    "es": "fotovoltaico pv placa solar modulo parque solar",
                    "pt": "fotovoltaico pv placa solar modulo usina solar",
                    "de": "solarmodul photovoltaik pv solaranlage dachsolar solarpark",
                    "it": "pannello solare fotovoltaico modulo fotovoltaico impianto solare parco fotovoltaico",
                    "nl": "zonnepaneel zonnepanelen pv fotovoltaïsch zonnepark zonnedak",
                    "pl": "panel fotowoltaiczny fotowoltaika panel słoneczny farma fotowoltaiczna park solarny",
                    "id": "panel surya fotovoltaik modul surya pembangkit surya",
                    "ja": "太陽光パネル 太陽電池 パネル ソーラーパネル メガソーラー",
                    "zh_CN": "太阳能板 光伏板 光伏组件 屋顶光伏 光伏电站 太阳能农场",
                    "zh_TW": "太陽能板 光電板 太陽能模組 屋頂太陽能 太陽能農場 光電場"},
    "swimming_pool": {"en": "pool pools swimming water",
                      "fr": "piscines bassin eau",
                      "es": "alberca pileta piscinas",
                      "pt": "piscinas tanque",
                      "de": "schwimmbecken pool schwimmbad wasserbecken",
                      "it": "piscina piscine vasca nuoto acqua",
                      "nl": "zwembad zwembaden zwemwater bad",
                      "pl": "basen pływalnia kąpielisko basen ogrodowy zbiornik kąpielowy",
                      "id": "kolam renang kolam berenang kolam air",
                      "ja": "プール 水泳場 スイミングプール 屋外プール 水場",
                      "zh_CN": "游泳池 泳池 水池 戏水池",
                      "zh_TW": "游泳池 水池 泳池 戲水池"},
}


def preset_search_terms(pid: str) -> dict[str, str]:
    """The extra search words for one preset, empty when it has none."""
    return dict(_SEARCH_TERMS.get(pid) or {})
