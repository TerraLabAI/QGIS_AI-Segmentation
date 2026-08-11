














from __future__ import annotations

_SEARCH_TERMS: dict[str, dict[str, str]] = {
    "building": {"en": "structure footprint block outline house roof rooftop silo home dwelling residential villa "
                       "roofs tiles surface building shingles grain granary farm",
                 "fr": "immeuble construction emprise batiment bati maison pavillon habitation villa toiture toit "
                       "toits couverture de bâtiment tuiles silo silos grenier",
                 "es": "inmueble construccion edificacion huella casa vivienda chalet casas residencial tejado techo "
                       "techos cubierta azotea techumbre tejas silo silos granero",
                 "pt": "imovel construcao edificacao casa moradia residencia casas telhado telhados cobertura laje "
                       "telhas teto predial silo silos celeiro",
                 "de": "bauwerk gebäude grundriss gebäudeumriss block haus wohnhaus wohngebäude villa dach dächer "
                       "dachziegel dachfläche dachumriss hausdach gebäudedach silo getreidesilo speicher "
                       "landwirtschaftssilo",
                 "it": "edificio struttura impronta sagoma casa abitazione residenza villa tetto tetti tegole del "
                       "copertura edilizia di silo cereali granaio silos agricolo",
                 "nl": "gebouw bebouwing pand omtrek contour huis woning woonhuis villa dak daken dakpannen dakrand "
                       "dakcontour dakvlak dakoppervlak huisdak gebou silo graansilo graanopslag landbouwsilo "
                       "opslagtank",
                 "pl": "budynek obrys budynku zabudowa konstrukcja dom mieszkalny willa siedlisko dach dachy "
                       "pokrycie dachowe dachu powierzchnia domu gonty silo silos elewator spichlerz zbiornik "
                       "zbożowy",
                 "id": "bangunan struktur tapak gedung konstruksi rumah hunian tempat tinggal vila atap genteng "
                       "permukaan garis silo biji-bijian lumbung penyimpanan hasil tani",
                 "ja": "建物 建築物 建物輪郭 建物フットプリント 住宅 家 一戸建て 民家 邸宅 屋根 屋上 瓦屋根 屋根形状 "
                       "屋根輪郭 建物屋根 屋根面 家屋根 屋根材 サイロ 穀物サイロ 貯蔵庫 農業サイロ 飼料サイロ",
                 "zh_CN": "建筑物 建筑 轮廓 建筑占地 房屋 住宅 民居 别墅 屋顶 房顶 瓦顶 屋顶轮廓 屋面 屋顶表面 "
                          "房屋屋顶 建筑屋顶 筒仓 粮仓 农业筒仓 储粮仓",
                 "zh_TW": "建築物 建物 建築輪廓 建物外框 建築占地 房屋 住宅 民宅 住家 別墅 屋頂 房頂 屋面 瓦頂 "
                          "屋頂輪廓 建築屋面 建築屋頂 瓦片屋頂 屋頂表面 穀倉 筒倉 農用筒倉 儲糧塔 儲料筒倉"},
    "tree": {"en": "trees canopy crown forest woodland fruit trees plantation grove tree rows apple orchard palm "
                   "tree palms coconut oilpalm palmtree datepalm",
             "fr": "arbres houppier canopee foret forêt verger arbres fruitiers plantation bosquet rangées arbres "
                   "pommeraie orchard palmier palmes cocotier huile palmeraie palmiers dattier",
             "es": "arboles copa dosel bosque frutales plantación huerto arboleda manzanos orchard huerto frutal "
                   "palmera palmas cocotero palma aceitera palmeral datilera",
             "pt": "arvores copa dossel floresta pomar árvores frutíferas plantação bosque fileiras árvores orchard "
                   "palmeira palmeiras coqueiro dendezeiro palmeiral tamareira palmas",
             "de": "bäume baumkrone kronendach wald gehölz obstgarten obstplantage baumgruppe baumreihen "
                   "apfelplantage orchard obstanlage palme palmen kokospalme ölpalme oelpalme palmenplantage palm",
             "it": "alberi chioma corona bosco foresta frutteto alberi da frutto piantagione filare di alberi meleti "
                   "filari orchard palma palme cocco oleifera palmeto palmetta",
             "nl": "boom bomen kruin kroon bos bosgebied boomgaard fruitbomen plantage boomgroep bomenrijen "
                   "appelboomgaard orchard palmboom palmen kokospalm oliepalm palmplantage palmgaard",
             "pl": "drzewo drzewa korona drzew las zagajnik sad drzewa owocowe plantacja gaj rzędy drzew sad "
                   "jabłoniowy orchard palma palmy kokosowa olejowa palm",
             "id": "pohon pepohonan tajuk mahkota hutan kebun buah perkebunan pohon buah rumpun barisan pohon "
                   "orchard palem kelapa sawit",
             "ja": "木 樹木 樹冠 林 森林 果樹園 果樹農園 樹木園 果樹列 りんご園 orchard ヤシの木 ヤシ ココナツ "
                   "アブラヤシ ヤシ園 椰子 パーム",
             "zh_CN": "树木 树冠 林地 森林 果园 果树园 植物园 林地 果树行 苹果园 orchard 棕榈树 棕榈 椰子树 油棕 "
                      "棕榈种植园 椰子",
             "zh_TW": "樹木 樹冠 林木 森林 林地 果園 果樹園 果樹種植地 林園 果樹列 蘋果園 orchard 棕櫚樹 棕櫚 椰子樹 "
                      "油棕 棕櫚種植園 椰子"},
    "road": {"en": "street highway motorway asphalt pavement traffic circle rotary junction roundabout",
             "fr": "rue chaussee autoroute voirie bitume giratoire carrefour rond point roundabout rond-point",
             "es": "calle via autopista pavimento asfalto glorieta redondel roundabout rotonda",
             "pt": "rua via rodovia asfalto pavimento rotula retorno roundabout rotatória",
             "de": "straße fahrbahn landstraße autobahn asphalt verkehrsweg kreisverkehr verkehrskreisel "
                   "kreisförmige kreuzung roundabout",
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
    "field": {"en": "fields cropland farmland parcel plot paddock crop farm field crop field agriculture arable "
                    "cultivated",
              "fr": "champs parcelle culture terrain lopin agricole champ terre cultures cultivée terres agricoles "
                    "zone",
              "es": "campos parcela cultivo terreno lote agrícola campo agricultura de sembrado labrado",
              "pt": "campos talhao parcela cultivo terreno lote talhão agrícola campo lavoura cultura agricultura "
                    "área de cultivado plantação",
              "de": "felder ackerland landwirtschaftsfläche parzelle weidefläche anbaufläche acker feld flurstück "
                    "agrarfläche kulturfläche bewirtschaftete fläche",
              "it": "campo campi terreno agricolo appezzamento parcella pascolo coltura coltivato particella "
                    "agricola agricoltura arabile seminativo azienda",
              "nl": "velden akker landbouwgrond perceel kavel weiland gewas boerenland landbouw landbouwperceel "
                    "bouwland gewasakker cultuurland",
              "pl": "pola pole uprawne grunty rolne parcela działka pastwisko uprawa rolna rolnictwo orne ziemia "
                    "uprawna gospodarstwo",
              "id": "lapangan lahan pertanian ladang petak lahan padang sawah kebun tanaman budidaya",
              "ja": "畑 耕地 農地 圃場 区画 放牧地 作付地 農業地 耕作地 作物畑 田畑",
              "zh_CN": "田地 耕地 农田 地块 地块区 牧场 作物田 农场地块 农业用地 种植地 农业地块 旱田",
              "zh_TW": "田地 農田 耕地 農地 地塊 園地 牧場 田區 作物地 農業用地 作物田 旱田 耕作地"},
    "water": {"en": "water body waterbody surface water pond basin wetland flood lake river stream reservoir stream "
                    "watercourse creek channel waterway",
              "fr": "eau plan d'eau etendue d'eau mare bassin zone humide lac riviere etang rivière fleuve cours "
                    "d'eau ruisseau canal chenal river",
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
              "pl": "woda zbiornik wodny akwen staw jezioro rzeka potok mokradło rzeka strumień ciek wodny potok "
                    "kanał koryto wodne droga wodna river",
              "id": "badan air permukaan air kolam danau sungai waduk rawa sungai anak sungai aliran air kanal "
                    "saluran air river",
              "ja": "水域 水面 池 湿地 湖 河川 貯水池 川 河川 小川 水路 渓流 河道 river",
              "zh_CN": "水体 水面 湖泊 河流 水塘 湿地 水库 河流 溪流 河道 水道 沟渠 水路 river",
              "zh_TW": "水域 水體 水面 池塘 湖泊 河流 濕地 溪流 河道 小溪 水道 河渠 水路 river"},
    "vegetation": {"en": "vegetated green cover greenery plants undergrowth scrub bush shrub hedge brush bushes "
                         "hedgerow row line boundary windbreak",
                   "fr": "vegetal couvert vegetal verdure plantes broussaille buisson arbuste fourre haie brise vent "
                         "arbustive alignement arbustes clôture végétale hedge",
                   "es": "cubierta vegetal verde plantas maleza arbusto matorral arbustos seto cercado lindero "
                         "cortavientos hedge",
                   "pt": "cobertura vegetal verde plantas mato arbusto arbustos moita cerca viva sebe fileira divisa "
                         "quebra vento hedge",
                   "de": "vegetation grünbewuchs begrünt pflanzen unterwuchs gebüsch strauch busch sträucher "
                         "unterholz hecke heckenreihe strauchreihe grenzhecke windschutzhecke hedge",
                   "it": "vegetazione copertura verde piante sottobosco macchia arbusti cespuglio arbusto cespugli "
                         "siepe filare barriera divisoria frangivento hedge",
                   "nl": "begroeiing vegetatie groen groenvoorziening planten ondergroei struweel struik struiken "
                         "kreupelhout bosjes haag heg heggen struikenrij haaglijn erfafscheiding windsingel hedge",
                   "pl": "roślinność zieleń pokrywa roślinna rośliny podszyt zarośla krzak krzew krzewy chaszcze "
                         "zakrzaczenia żywopłot szpaler krzewów granica żywopłotowa pas zarośli wiatrochron hedge",
                   "id": "vegetasi tutupan hijau tanaman semak tumbuhan bawah belukar perdu rimbunan pagar berbaris "
                         "batas penahan angin hedge",
                   "ja": "植生 緑地 緑被 植物 下草 藪 灌木 低木 茂み ブラッシュ 植え込み 生垣 垣根 低木列 境界樹 "
                         "防風林 hedge",
                   "zh_CN": "植被 绿化 植物 覆盖 下层植被 灌丛 灌木 灌木丛 荒草 树篱 灌木篱 篱笆绿篱 绿篱带 边界绿篱 "
                            "防风林带 hedge",
                   "zh_TW": "植被 綠色覆蓋 綠化 植物 下層植被 灌叢 灌木 灌木叢 草叢 矮樹 樹籬 灌木籬 樹籬線 邊界樹籬 "
                            "防風林帶 綠籬 hedge"},
    "vehicle": {"en": "car, cars, vehicles, auto, automobile, van, truck, vehicle, lorry, semi, hgv, freight",
                "fr": "voiture voitures vehicule vehicules automobile camionnette camion poids lourd semi-remorque "
                      "fourgon",
                "es": "coche coches vehiculo vehiculos automovil furgoneta autos camión camiones trailer furgon",
                "pt": "carro carros veiculo veiculos automovel van caminhão caminhoes carreta furgao",
                "de": "auto autos fahrzeuge pkw wagen transporter fahrzeug lastwagen lkw sattelschlepper "
                      "schwerlastwagen lieferwagen",
                "it": "auto automobile vettura veicoli macchina veicolo camion autocarro tir autotreno furgone",
                "nl": "auto autos voertuigen personenauto wagen bestelbus voertuig vrachtwagen truck trekker "
                      "oplegger bestelwagen",
                "pl": "samochód samochody pojazdy auto furgonetka pojazd ciężarówka tir ciężarowy zestaw furgon",
                "id": "mobil kendaraan otomobil van truk lori angkutan trailer",
                "ja": "車 自動車 乗用車 車両 バン トラック 貨物車 大型車 トレーラー ローリー",
                "zh_CN": "汽车 轿车 小汽车 车辆 面包车 卡车 货车 重型卡车 半挂车 厢式货车",
                "zh_TW": "汽車 車輛 小客車 轎車 廂型車 卡車 貨車 拖車 大型貨車 運輸車"},
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
    "parking_lot": {"en": "car park, parking, parking space",
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
}


def preset_search_terms(pid: str) -> dict[str, str]:

    return dict(_SEARCH_TERMS.get(pid) or {})
