# 🎯 OPTIMAL AI MAPPING PROMPT - COMPREHENSIVE GEMINI TESTING

**PURPOSE**: This document contains the most optimal, descriptive prompt structure for direct testing with Gemini AI chat interface. Contains ALL necessary data files in their complete, unaltered form.

**USAGE**: Copy the entire prompt section below and paste it directly into Gemini AI chat for testing and optimization.

---

## 🚀 COMPREHENSIVE AI MAPPING PROMPT FOR GEMINI

### **MISSION STATEMENT**
You are an expert e-commerce data transformation specialist with deep knowledge of German marketplace operations and Amazon's complex product taxonomy. Your task is to perform a sophisticated mapping of authentic German workwear product data to Amazon's stringent marketplace format, ensuring complete compliance with all field requirements and maintaining the authentic product characteristics.

### **CRITICAL REQUIREMENTS**
1. **ZERO TRUNCATION**: Process ALL product variants - every single variant must be mapped
2. **AUTHENTIC DATA**: Use only real source values from the compressed product data - NO placeholder or mock data
3. **COMPLETE FIELD COVERAGE**: Map all mandatory Amazon marketplace fields
4. **VALIDATION COMPLIANCE**: Ensure all field values comply with Amazon's validation rules and allowed values
5. **INHERITANCE ACCURACY**: Properly implement parent-child field inheritance as defined in the template
6. **BUSINESS LOGIC**: Apply intelligent mapping decisions based on product characteristics and marketplace standards
7. **OUTPUT LANGUAGE**: German

---

## 📋 COMPLETE TEMPLATE SPECIFICATION (step4_template.json)

```json
{
  "template_structure": {
    "parent_product": {
      "fields": {
        "feed_product_type": {
          "display_name": "Produkttyp",
          "data_type": "string",
          "constraints": {
            "value_count": 1,
            "max_length": 5
          },
          "applies_to_children": true,
          "validation_rules": {
            "required": true,
            "data_type": "string",
            "max_length": 5,
            "allowed_values": [
              "pants"
            ]
          }
        },
        "brand_name": {
          "display_name": "Marke",
          "data_type": "string",
          "constraints": {
            "value_count": 3,
            "max_length": 42
          },
          "applies_to_children": true,
          "validation_rules": {
            "required": true,
            "data_type": "string",
            "max_length": 42,
            "allowed_values": [
              "TMG Clothing",
              "TMG INTERNATIONAL Textile Management Group",
              "TMG"
            ]
          }
        },
        "external_product_id": {
          "display_name": "Hersteller-Barcode",
          "data_type": "string",
          "constraints": {
            "value_count": 0,
            "max_length": null
          },
          "applies_to_children": false,
          "validation_rules": {
            "required": false,
            "data_type": "string"
          }
        },
        "external_product_id_type": {
          "display_name": "Barcode-Typ",
          "data_type": "string",
          "constraints": {
            "value_count": 5,
            "max_length": 4
          },
          "applies_to_children": true,
          "validation_rules": {
            "required": true,
            "data_type": "string",
            "max_length": 4,
            "allowed_values": [
              "EAN",
              "UPC",
              "ASIN",
              "GTIN",
              "GCID"
            ]
          }
        },
        "item_name": {
          "display_name": "Produktname",
          "data_type": "string",
          "constraints": {
            "value_count": 0,
            "max_length": null
          },
          "applies_to_children": false,
          "validation_rules": {
            "required": false,
            "data_type": "string"
          }
        },
        "recommended_browse_nodes": {
          "display_name": "Produktkategorisierung (Suchpfad)",
          "data_type": "numeric",
          "constraints": {
            "value_count": 1,
            "max_length": 10
          },
          "applies_to_children": true,
          "validation_rules": {
            "required": true,
            "data_type": "numeric",
            "max_length": 10,
            "allowed_values": [
              "1981663031"
            ]
          }
        },
        "standard_price": {
          "display_name": "Preis",
          "data_type": "string",
          "constraints": {
            "value_count": 0,
            "max_length": null
          },
          "applies_to_children": false,
          "validation_rules": {
            "required": false,
            "data_type": "string"
          }
        },
        "quantity": {
          "display_name": "Anzahl",
          "data_type": "string",
          "constraints": {
            "value_count": 0,
            "max_length": null
          },
          "applies_to_children": false,
          "validation_rules": {
            "required": false,
            "data_type": "string"
          }
        },
        "target_gender": {
          "display_name": "Geschlecht",
          "data_type": "string",
          "constraints": {
            "value_count": 3,
            "max_length": 8
          },
          "applies_to_children": true,
          "validation_rules": {
            "required": true,
            "data_type": "string",
            "max_length": 8,
            "allowed_values": [
              "Unisex",
              "Weiblich",
              "Männlich"
            ]
          }
        },
        "age_range_description": {
          "display_name": "Altersgruppe Beschreibung",
          "data_type": "string",
          "constraints": {
            "value_count": 4,
            "max_length": 11
          },
          "applies_to_children": true,
          "validation_rules": {
            "required": true,
            "data_type": "string",
            "max_length": 11,
            "allowed_values": [
              "Baby",
              "Kind",
              "Erwachsener",
              "Kleinkind"
            ]
          }
        },
        "main_image_url": {
          "display_name": "URL des Hauptbilds",
          "data_type": "string",
          "constraints": {
            "value_count": 0,
            "max_length": null
          },
          "applies_to_children": false,
          "validation_rules": {
            "required": false,
            "data_type": "string"
          }
        },
        "department_name": {
          "display_name": "Name der Abteilung",
          "data_type": "string",
          "constraints": {
            "value_count": 9,
            "max_length": 14
          },
          "applies_to_children": false,
          "validation_rules": {
            "required": true,
            "data_type": "string",
            "max_length": 14,
            "allowed_values": [
              "Herren",
              "Baby - Jungen",
              "Unisex Baby",
              "Baby - Mädchen",
              "Damen",
              "Jungen",
              "Unisex Kinder",
              "Mädchen",
              "Unisex"
            ]
          }
        },
        "fabric_type": {
          "display_name": "Gewebetyp",
          "data_type": "string",
          "constraints": {
            "value_count": 0,
            "max_length": null
          },
          "applies_to_children": false,
          "validation_rules": {
            "required": false,
            "data_type": "string"
          }
        },
        "list_price_with_tax": {
          "display_name": "Preis mit Steuern zur Anzeige",
          "data_type": "string",
          "constraints": {
            "value_count": 0,
            "max_length": null
          },
          "applies_to_children": false,
          "validation_rules": {
            "required": false,
            "data_type": "string"
          }
        }
      },
      "field_count": 14,
      "required_fields": [
        "feed_product_type",
        "brand_name",
        "external_product_id_type",
        "recommended_browse_nodes",
        "target_gender",
        "age_range_description",
        "department_name"
      ]
    },
    "child_variants": {
      "fields": {
        "outer_material_type": {
          "display_name": "Hauptmaterial",
          "data_type": "string",
          "constraints": {
            "value_count": 33,
            "max_length": 17
          },
          "variation_type": "material",
          "validation_rules": {
            "required": false,
            "data_type": "string",
            "max_length": 17,
            "allowed_values": [
              "Wolle",
              "Baumwolle",
              "Wildleder",
              "Fur",
              "Lackleder",
              "Samt",
              "Fleece",
              "Pelzimitat",
              "Merino",
              "Angorawolle",
              "Angora",
              "Synthetik",
              "Leinen",
              "Alpaka",
              "Gummi",
              "Mohair",
              "Paillettenbesetzt",
              "Leder",
              "Glattleder",
              "Filz",
              "Pelz",
              "Jeans",
              "Fellimitat",
              "Cord",
              "Satin",
              "Kaschmir",
              "Kashmir",
              "Synthetisch",
              "Daunen",
              "Pailletten",
              "Hanf",
              "Seide",
              "Denim"
            ]
          }
        },
        "bottoms_size_system": {
          "display_name": "Größensystem des Artikels",
          "data_type": "string",
          "constraints": {
            "value_count": 1,
            "max_length": 17
          },
          "variation_type": "size",
          "validation_rules": {
            "required": true,
            "data_type": "string",
            "max_length": 17,
            "allowed_values": [
              "DE / NL / SE / PL"
            ]
          }
        },
        "bottoms_size_class": {
          "display_name": "Größenklassifizierung",
          "data_type": "string",
          "constraints": {
            "value_count": 5,
            "max_length": 24
          },
          "variation_type": "size",
          "validation_rules": {
            "required": true,
            "data_type": "string",
            "max_length": 24,
            "allowed_values": [
              "Numerisch",
              "Bundweite & Schrittlänge",
              "Bundweite",
              "Alphanumerisch",
              "Alter"
            ]
          }
        },
        "color_map": {
          "display_name": "Farbfamilie",
          "data_type": "string",
          "constraints": {
            "value_count": 22,
            "max_length": 12
          },
          "variation_type": "attribute",
          "validation_rules": {
            "required": false,
            "data_type": "string",
            "max_length": 12,
            "allowed_values": [
              "Mehrfarbig",
              "Braun",
              "Blau",
              "Bronze",
              "Orange",
              "Rosa",
              "Schwarz",
              "Cremefarben",
              "Beige",
              "Weiß",
              "Violett",
              "Gold",
              "Metallisch",
              "Grau",
              "Türkis",
              "Durchsichtig",
              "Grün",
              "Lila",
              "Rot",
              "Elfenbein",
              "Silber",
              "Gelb"
            ]
          }
        },
        "color_name": {
          "display_name": "Farbe",
          "data_type": "string",
          "constraints": {
            "value_count": 0,
            "max_length": null
          },
          "variation_type": "color",
          "validation_rules": {
            "required": false,
            "data_type": "string"
          }
        },
        "size_name": {
          "display_name": "Größe",
          "data_type": "string",
          "constraints": {
            "value_count": 0,
            "max_length": null
          },
          "variation_type": "size",
          "validation_rules": {
            "required": false,
            "data_type": "string"
          }
        },
        "size_map": {
          "display_name": "Größenzuordnung",
          "data_type": "numeric",
          "constraints": {
            "value_count": 40,
            "max_length": 8
          },
          "variation_type": "size",
          "validation_rules": {
            "required": false,
            "data_type": "numeric",
            "max_length": 8,
            "allowed_values": [
              "48",
              "27",
              "47",
              "29",
              "34",
              "58",
              "64",
              "40",
              "26",
              "35",
              "42",
              "66",
              "56",
              "30",
              "23",
              "43",
              "22",
              "49",
              "25",
              "37",
              "28",
              "32",
              "36",
              "44",
              "38",
              "50",
              "24",
              "68",
              "62",
              "33",
              "One Size",
              "70",
              "31",
              "45",
              "41",
              "46",
              "39",
              "60",
              "52",
              "54"
            ]
          }
        },
        "country_of_origin": {
          "display_name": "Land/Region der Herkunft",
          "data_type": "string",
          "constraints": {
            "value_count": 268,
            "max_length": 33
          },
          "variation_type": "attribute",
          "validation_rules": {
            "required": false,
            "data_type": "string",
            "max_length": 33,
            "allowed_values": [
              "Cook-Inseln",
              "Simbabwe",
              "Malta",
              "Ecuador",
              "St. Pierre und Miquelon",
              "Slowakei",
              "Tansania",
              "Amerikanisch-Samoa",
              "Norwegen",
              "Demokratische Republik Kongo",
              "Swasiland",
              "Costa Rica",
              "Falkland-Inseln",
              "Lesotho",
              "Katar",
              "Südsudan",
              "Nordkorea",
              "Mauretanien",
              "Niger",
              "Zaire",
              "British Virgin Islands",
              "Türkei",
              "Serbien und Montenegro",
              "Tunesien",
              "Vereinigte Arabische Emirate",
              "WD",
              "Armenien",
              "Jemen",
              "São Tomé und Príncipe",
              "Saint Lucia",
              "Svalbard",
              "Angola",
              "Haiti",
              "Grenada",
              "Namibia",
              "Republik Kongo",
              "Unbekannt",
              "Belgien",
              "Salomon-Inseln",
              "Algerien",
              "Jugoslawien",
              "Bhutan",
              "Bahamas",
              "Grönland",
              "Kambodscha",
              "Usbekistan",
              "XC",
              "Mali",
              "Guadeloupe",
              "Niederländische Antillen",
              "Kasachstan",
              "Niederlande",
              "Guinea-Bissau",
              "Christmas Island",
              "XN",
              "Burkina Faso",
              "Mayotte",
              "Bosnien und Herzegowina",
              "Bermuda",
              "Jersey",
              "Russland",
              "Jamaika",
              "Panama",
              "Argentinien",
              "Französisch Südliche Territorien",
              "Portugal",
              "Tonga",
              "Ascension",
              "Libanon",
              "Kanada",
              "Bahrein",
              "Elfenbeinküste",
              "Oman",
              "Philippinen",
              "Puerto Rico",
              "St. Kitts und Nevis",
              "Wallis und Futuna",
              "Mosambik",
              "Westsahara",
              "Saint Helena",
              "Cayman Islands",
              "Cocos (Keeling) Inseln",
              "Heiliger Stuhl (Vatikanstadt)",
              "Sambia",
              "Estland",
              "Fidschi",
              "Tokelau",
              "Kolumbien",
              "Burundi",
              "Färöer-Inseln",
              "Liechtenstein",
              "Luxemburg",
              "Benin",
              "Nigeria",
              "Paraguay",
              "Turks-und Caicosinseln",
              "Macau",
              "Ghana",
              "El Salvador",
              "Tschechische Republik",
              "Guinea",
              "Chile",
              "Israel",
              "Hongkong",
              "Vereinigte Staaten",
              "Zentralafrikanische Republik",
              "Malawi",
              "Bangladesch",
              "Malaysia",
              "Suriname",
              "Irland",
              "Ruanda",
              "Uruguay",
              "Slowenien",
              "Syrien",
              "Neuseeland",
              "Saint-Martin",
              "Tadschikistan",
              "Isle of Man",
              "Dschibuti",
              "Tschad",
              "Südkorea",
              "Nauru",
              "Schweiz",
              "Kiribati",
              "Mauritius",
              "Litauen",
              "Palau",
              "Réunion",
              "Vereinigtes Königreich",
              "Bouvet-Insel",
              "Malediven",
              "Weißrussland",
              "Aland-Inseln",
              "Montserrat",
              "Spanien",
              "Niue",
              "Äthiopien",
              "Norfolk Island",
              "Kroatien",
              "Neukaledonien",
              "XY",
              "Guatemala",
              "Ungarn",
              "Dänemark",
              "Marshall-Inseln",
              "Polen",
              "Belize",
              "Albanien",
              "Aserbaidschan",
              "Bolivien",
              "Guyana",
              "Kirgisistan",
              "Kamerun",
              "Griechenland",
              "Saudi-Arabien",
              "Palästinensische Autonomiegebiete",
              "Marokko",
              "Gabun",
              "Madagaskar",
              "Rumänien",
              "Thailand",
              "Togo",
              "Moldawien",
              "Serbien",
              "China",
              "Lettland",
              "Aruba",
              "Vanuatu",
              "Barbados",
              "Frankreich",
              "Dominica",
              "Pakistan",
              "Nicaragua",
              "Northern Mariana Islands",
              "Island",
              "Mexiko",
              "St. Barthélemy",
              "Gambia",
              "Mikronesien",
              "Jordanien",
              "Antigua und Barbuda",
              "Kap Verde",
              "Südafrika",
              "Venezuela",
              "Guam",
              "Ukraine",
              "Japan",
              "Iran",
              "Afghanistan",
              "Trinidad und Tobago",
              "Burma (Myanmar)",
              "San Marino",
              "Komoren",
              "WZ",
              "US Virgin Islands",
              "XB",
              "Senegal",
              "Taiwan",
              "Turkmenistan",
              "Deutschland",
              "Kuba",
              "Pitcairn-Inseln",
              "Martinique",
              "Französisch-Polynesien",
              "Saint Vincent und die Grenadinen",
              "Samoa",
              "Singapur",
              "XE",
              "XM",
              "Nepal",
              "Irak",
              "Kanarische Inseln",
              "Zypern",
              "S. Georgia und S. Sandwich ISLs.",
              "Bonaire, St. Eustatius und Saba",
              "Anguilla",
              "Australien",
              "Kenia",
              "Liberia",
              "Mongolei",
              "Schweden",
              "Georgien",
              "Eritrea",
              "Montenegro",
              "Italien",
              "Sudan",
              "Brunei",
              "Heard und McDonald Inseln",
              "Laos",
              "Tuvalu",
              "Andorra",
              "Tristan da Cunha",
              "Uganda",
              "Seychellen",
              "Honduras",
              "Guernsey",
              "Österreich",
              "Papua-Neuguinea",
              "Vietnam",
              "Sri Lanka",
              "Bulgarien",
              "Kuwait",
              "Gibraltar",
              "US-Ozeanien",
              "XK",
              "Brasilien",
              "Curaçao",
              "Indien",
              "Peru",
              "St. Martin (französischer Teil)",
              "Monaco",
              "Botswana",
              "Finnland",
              "Indonesien",
              "Großbritannien",
              "Osttimor",
              "British Indian Ocean Territory",
              "Dominikanische Republik",
              "Libyen",
              "Französisch-Guayana",
              "Nordmazedonien",
              "Ägypten",
              "Äquatorial-Guinea",
              "Sierra Leone",
              "Somalia",
              "Timor-Leste",
              "Antarktika"
            ]
          }
        },
        "item_sku": {
          "display_name": "Verkäufer-SKU",
          "data_type": "string",
          "constraints": {
            "value_count": 0,
            "max_length": null
          },
          "variation_type": "attribute",
          "validation_rules": {
            "required": false,
            "data_type": "string"
          }
        }
      },
      "field_count": 9,
      "variable_fields": [
        "outer_material_type",
        "color_map",
        "color_name",
        "size_name",
        "size_map",
        "country_of_origin",
        "item_sku"
      ],
      "inherited_fields": [
        "outer_material_type",
        "bottoms_size_system",
        "bottoms_size_class",
        "color_map",
        "color_name",
        "size_name",
        "size_map",
        "country_of_origin"
      ]
    },
    "field_relationships": {
      "parent_defines": [
        {
          "field": "feed_product_type",
          "inheritance_type": "mandatory",
          "override_allowed": false
        },
        {
          "field": "brand_name",
          "inheritance_type": "mandatory",
          "override_allowed": false
        },
        {
          "field": "external_product_id_type",
          "inheritance_type": "mandatory",
          "override_allowed": false
        },
        {
          "field": "recommended_browse_nodes",
          "inheritance_type": "mandatory",
          "override_allowed": false
        },
        {
          "field": "target_gender",
          "inheritance_type": "mandatory",
          "override_allowed": false
        },
        {
          "field": "age_range_description",
          "inheritance_type": "mandatory",
          "override_allowed": false
        }
      ],
      "variant_overrides": [
        {
          "field": "outer_material_type",
          "default_source": "parent",
          "variation_required": true
        },
        {
          "field": "bottoms_size_system",
          "default_source": "parent",
          "variation_required": false
        },
        {
          "field": "bottoms_size_class",
          "default_source": "parent",
          "variation_required": false
        },
        {
          "field": "color_map",
          "default_source": "parent",
          "variation_required": true
        },
        {
          "field": "color_name",
          "default_source": "parent",
          "variation_required": true
        },
        {
          "field": "size_name",
          "default_source": "parent",
          "variation_required": true
        },
        {
          "field": "size_map",
          "default_source": "parent",
          "variation_required": true
        },
        {
          "field": "country_of_origin",
          "default_source": "parent",
          "variation_required": true
        }
      ],
      "shared_constraints": {}
    }
  },
  "usage_instructions": {
    "description": "Template for structured parent-child product mapping",
    "parent_product_usage": "Define shared characteristics and product family",
    "child_variants_usage": "Define variable attributes and specific variants",
    "inheritance_rules": "Children inherit parent values unless overridden"
  }
}
```

---

## 📊 COMPLETE SOURCE PRODUCT DATA (step2_compressed.json)

```json
{
  "parent_data": {
    "FVALUE_3_6": "Goliath-Cord",
    "FNAME_3_5": "Serie",
    "ORDER_UNIT": "C62",
    "FVALUE_3_10": "Messing/Gold",
    "FVALUE_3_23": "von außen erreichbar",
    "FNAME_2_1": "-",
    "CUSTOMS_TARIFF_NUMBER": 62034211,
    "MIN_QUANTITY": 1,
    "FVALUE_3_25": "Die Gesäßnaht wird nach der Methode einer Anzughose gefertigt, was eine individuelle Anpassung der Hose ermöglicht.",
    "FVALUE_3_20": "ja",
    "FVALUE_3_11": "schwarz",
    "FNAME_3_1": "Farbcode",
    "MIME_DESCR_3": "Logo",
    "MANUFACTURER_NAME": "EIKO",
    "FNAME_3_20": "Knietaschen für Kniepolster",
    "FNAME_3_22": "Einschub Knietaschen",
    "FVALUE_2_2": "mit Echtlederbesatz",
    "FNAME_3_25": "Gesäßnaht",
    "FVALUE_3_12": "zwei eingesetzte Schubtaschen",
    "FNAME_3_16": "Taschenpaspelierung",
    "FVALUE_3_24": "Eine dreifache Kappnaht bietet erhöhte Festigkeit und Haltbarkeit, ideal für stark beanspruchte Bereiche. Sie ist elastischer als eine normale Naht, da sich der Faden bei Belastung längen kann. Zudem sorgt sie für ein sauberes und flaches Finish.",
    "FNAME_3_18": "Knöpfe für Hosenträger",
    "MANUFACTURER_PID": 41282,
    "MIME_DESCR_1": "Hauptbild",
    "CONTENT_UNIT": "C62",
    "FVALUE_3_9": "9 mm",
    "FVALUE_2_1": "Zunfthose",
    "FNAME_3_15": "Seitentasche oben rechts",
    "SYSTEMNAME_2": "udf_NMTOPFEATURES-1.0",
    "FNAME_3_24": "Schrittnaht",
    "MANUFACTURER_TYPE_DESCRIPTION": "PERCY",
    "INTERVAL_QUANTITY": 1,
    "FVALUE_3_3": "Schwarz",
    "MASTER": 41282,
    "FNAME_3_4": "Fußweite",
    "PRODUCT_STATUS": "ACTIVE",
    "FNAME_3_2": "Größe",
    "FVALUE_3_18": "6 Knöpfe um Hosenträger mit Patten oder Biesen zu befestigen",
    "FNAME_3_14": "Seitentasche oben links",
    "MIME_PURPOSE_2": "URL",
    "FNAME_3_6": "Oberstoff",
    "FVALUE_3_8": "100% Baumwolle mit EIKO Logo",
    "FUNIT_3_4": "cm",
    "FNAME_3_11": "Reißverschluss Gewebe Farbe",
    "FNAME_3_9": "Reißverschlussbreite",
    "FNAME_3_13": "Gesäßtaschen",
    "FVALUE_3_21": "Oberstoff",
    "MIME_THUMB_2": "https://blob.redpim.de/company-53e006db-2b74-4ce1-5a4d-08dca19c0e21/mimes/_normal.jpg",
    "FNAME_3_12": "Schubtaschen vorn",
    "MIME_THUMB_1": "https://blob.redpim.de/company-53e006db-2b74-4ce1-5a4d-08dca19c0e21/mimes/4160633_6A261AB71579891EE1DFFB78F85DE71405A04C3B7A6038B2C33C4D5B4B640F52_normal.jpg",
    "FNAME_3_10": "Reißverschluss Zähne Farbe",
    "FNAME_3_19": "Taschenverstärkung",
    "FNAME_3_3": "Farbe",
    "DESCRIPTION_LONG": "Diese Zunfthose mit normaler Fußweite sorgt für ein meisterliches Auftreten auf dem Bau oder bei der Repräsentation. Echtlederpaspel und Echtlederecken geben der Dreidrahtcord-Hose eine besonders edle Optik. Durch die klassische Herrenkonfektionierung lässt sich die Hose an der Gesäßnaht in der Größe variieren.",
    "MIME_PURPOSE_1": "Normal",
    "SYSTEMNAME_1": "udf_NMMARKETINGCLAIM-1.0",
    "FVALUE_3_17": "Flache Nähte für reibungslosen Komfort und erhöhte Stabilität",
    "SYSTEMNAME_3": "udf_NMTECHNICALDETAILS-1.0",
    "FNAME_3_23": "Ausführung Knietasche",
    "FNAME_3_7": "Taschenfutter",
    "COUNTRY_OF_ORIGIN": "Tunesien",
    "FVALUE_3_16": "Cordura",
    "FNAME_3_8": "Bundband",
    "FVALUE_3_22": "von unten - mit Klettverschluss",
    "FVALUE_3_5": "ZUNFT EXCLUSIV",
    "FNAME_3_17": "Keil im Schritt",
    "FVALUE_3_1": 40,
    "MIME_PURPOSE_3": "Logo",
    "FVALUE_3_7": "100% Polyester extra stark & schwer",
    "FVALUE_3_13": "eine Gesäßtasche rechts mit Lasche und Knopfverschluss",
    "NOCUPEROU": 1,
    "FNAME_3_21": "Material Knietaschen",
    "FVALUE_3_19": "rechteckige Verstärkungen aus echtem Vollrindleder",
    "WEIGHT": "1,14",
    "MIME_THUMB_3": "https://blob.redpim.de/company-53e006db-2b74-4ce1-5a4d-08dca19c0e21/mimes/326486_404DFA136769374B5D54953119A6A2E3380034E6ED0EFB67D648AC04236DAB49_normal.jpg",
    "FNAME_2_2": "-",
    "FVALUE_3_14": "extra breite Leistentasche",
    "GROUP_STRING": "Root|Zunftbekleidung|Zunfthosen",
    "MIME_DESCR_2": "Link",
    "FVALUE_3_15": "extra breite Leistentasche",
    "_parent_sku": "41282",
    "_child_count": 28,
    "_analysis_timestamp": 1757021755.226917
  },
  "data_rows": [
    {
      "SUPPLIER_PID": "41282_40_44",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 44",
      "INTERNATIONAL_PID": 4033976076973,
      "FVALUE_3_2": 44
    },
    {
      "SUPPLIER_PID": "41282_40_46",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 46",
      "INTERNATIONAL_PID": 4033976076980,
      "FVALUE_3_2": 46
    },
    {
      "SUPPLIER_PID": "41282_40_48",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 48",
      "INTERNATIONAL_PID": 4033976076997,
      "FVALUE_3_2": 48
    },
    {
      "SUPPLIER_PID": "41282_40_50",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 50",
      "INTERNATIONAL_PID": 4033976077000,
      "FVALUE_3_2": 50
    },
    {
      "SUPPLIER_PID": "41282_40_52",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 52",
      "INTERNATIONAL_PID": 4033976077017,
      "FVALUE_3_2": 52
    },
    {
      "SUPPLIER_PID": "41282_40_54",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 54",
      "INTERNATIONAL_PID": 4033976077024,
      "FVALUE_3_2": 54
    },
    {
      "SUPPLIER_PID": "41282_40_56",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 56",
      "INTERNATIONAL_PID": 4033976077031,
      "FVALUE_3_2": 56
    },
    {
      "SUPPLIER_PID": "41282_40_58",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 58",
      "INTERNATIONAL_PID": 4033976077048,
      "FVALUE_3_2": 58
    },
    {
      "SUPPLIER_PID": "41282_40_60",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 60",
      "INTERNATIONAL_PID": 4033976077055,
      "FVALUE_3_2": 60
    },
    {
      "SUPPLIER_PID": "41282_40_62",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 62",
      "INTERNATIONAL_PID": 4033976077062,
      "FVALUE_3_2": 62
    },
    {
      "SUPPLIER_PID": "41282_40_64",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 64",
      "INTERNATIONAL_PID": 4033976077079,
      "FVALUE_3_2": 64
    },
    {
      "SUPPLIER_PID": "41282_40_66",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 66",
      "INTERNATIONAL_PID": 4033976077086,
      "FVALUE_3_2": 66
    },
    {
      "SUPPLIER_PID": "41282_40_102",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 102",
      "INTERNATIONAL_PID": 4033976077123,
      "FVALUE_3_2": 102
    },
    {
      "SUPPLIER_PID": "41282_40_106",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 106",
      "INTERNATIONAL_PID": 4033976077130,
      "FVALUE_3_2": 106
    },
    {
      "SUPPLIER_PID": "41282_40_110",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 110",
      "INTERNATIONAL_PID": 4033976077147,
      "FVALUE_3_2": 110
    },
    {
      "SUPPLIER_PID": "41282_40_114",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 114",
      "INTERNATIONAL_PID": 4033976077154,
      "FVALUE_3_2": 114
    },
    {
      "SUPPLIER_PID": "41282_40_90",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 90",
      "INTERNATIONAL_PID": 4033976077093,
      "FVALUE_3_2": 90
    },
    {
      "SUPPLIER_PID": "41282_40_94",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 94",
      "INTERNATIONAL_PID": 4033976077109,
      "FVALUE_3_2": 94
    },
    {
      "SUPPLIER_PID": "41282_40_98",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 98",
      "INTERNATIONAL_PID": 4033976077116,
      "FVALUE_3_2": 98
    },
    {
      "SUPPLIER_PID": "41282_40_23",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 23",
      "INTERNATIONAL_PID": 4033976076881,
      "FVALUE_3_2": 23
    },
    {
      "SUPPLIER_PID": "41282_40_24",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 24",
      "INTERNATIONAL_PID": 4033976076898,
      "FVALUE_3_2": 24
    },
    {
      "SUPPLIER_PID": "41282_40_25",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 25",
      "INTERNATIONAL_PID": 4033976076904,
      "FVALUE_3_2": 25
    },
    {
      "SUPPLIER_PID": "41282_40_26",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 26",
      "INTERNATIONAL_PID": 4033976076911,
      "FVALUE_3_2": 26
    },
    {
      "SUPPLIER_PID": "41282_40_27",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 27",
      "INTERNATIONAL_PID": 4033976076928,
      "FVALUE_3_2": 27
    },
    {
      "SUPPLIER_PID": "41282_40_28",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 28",
      "INTERNATIONAL_PID": 4033976076935,
      "FVALUE_3_2": 28
    },
    {
      "SUPPLIER_PID": "41282_40_29",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 29",
      "INTERNATIONAL_PID": 4033976076942,
      "FVALUE_3_2": 29
    },
    {
      "SUPPLIER_PID": "41282_40_30",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 30",
      "INTERNATIONAL_PID": 4033976076959,
      "FVALUE_3_2": 30
    },
    {
      "SUPPLIER_PID": "41282_40_31",
      "DESCRIPTION_SHORT": "PERCY - Zunfthose - Serie ZUNFT EXCLUSIV - Goliath-Cord - mit Echtlederbesatz - Schwarz - Größe: 31",
      "INTERNATIONAL_PID": 4033976076966,
      "FVALUE_3_2": 31
    }
  ]
}
```

---

## 🎨 COMPLETE STRUCTURE EXAMPLE AS EXPECTED OUTPUT FORMAT (step4_1_structure_example.json)

```json
{
  "field_categorization": {
    "coverage_validation": {
      "coverage_complete": true,
      "expected_total": 23,
      "parent_coverage": 14,
      "variant_coverage": 9
    },
    "parent_fields": [
      "age_range_description",
      "bottoms_size_class",
      "bottoms_size_system",
      "brand_name",
      "country_of_origin",
      "department_name",
      "external_product_id_type",
      "fabric_type",
      "feed_product_type",
      "item_name",
      "main_image_url",
      "outer_material_type",
      "recommended_browse_nodes",
      "target_gender"
    ],
    "total_fields": 23,
    "variant_fields": [
      "color_map",
      "color_name",
      "external_product_id",
      "item_sku",
      "list_price_with_tax",
      "quantity",
      "size_map",
      "size_name",
      "standard_price"
    ]
  },
  "generation_timestamp": "2025-09-04T23:37:23.443073Z",
  "mandatory_field_coverage": "23/23",
  "parent_data": {
    "age_range_description": "Kleinkind",
    "bottoms_size_class": "Bundweite & Schrittlänge",
    "bottoms_size_system": "DE / NL / SE / PL",
    "brand_name": "TMG Clothing",
    "country_of_origin": "Mauritius",
    "department_name": "Unisex",
    "external_product_id_type": "EAN",
    "fabric_type": "Cotton",
    "feed_product_type": "pants",
    "item_name": "PERCY Zunfthose",
    "main_image_url": "https://example.com/image.jpg",
    "outer_material_type": "Leder",
    "recommended_browse_nodes": "1981663031",
    "target_gender": "Männlich"
  },
  "structure_version": "1.0",
  "variants": [
    {
      "variant_1": {
        "color_map": "Weiß",
        "color_name": "Schwarz",
        "external_product_id": "4033976004549",
        "item_sku": "41282_40_44",
        "list_price_with_tax": "59.49",
        "quantity": "10",
        "size_map": "One Size",
        "size_name": "44",
        "standard_price": "49.99"
      }
    },
    {
      "variant_2": {
        "color_map": "Weiß",
        "color_name": "Schwarz",
        "external_product_id": "4033976004556",
        "item_sku": "41282_40_46",
        "list_price_with_tax": "59.49",
        "quantity": "8",
        "size_map": "46",
        "size_name": "46",
        "standard_price": "49.99"
      }
    }
  ]
}
```

---

## 🧠 COMPREHENSIVE MAPPING INSTRUCTIONS

### **Your Mission**
Transform the authentic German workwear product data into a complete Amazon marketplace JSON format that:
1. Maps ALL product variants
2. Populates all mandatory fields using real source data
3. Follows the exact structure from the step4_1_structure_example.json
4. Complies with all validation rules from step4_template.json
5. Applies intelligent business logic for field derivation

---

## EXECUTION COMMAND

**GENERATE THE COMPLETE AMAZON MARKETPLACE JSON NOW**

Using the three complete data files provided above:
1. The step4_template.json (validation rules and allowed values)
2. The step2_compressed.json (real product source data)  
3. The step4_1_structure_example.json (output format structure)

Create a comprehensive JSON transformation that:
- Processes ALL variants without any truncation
- Map all 23 mandatory fields using authentic source data
- Follow the exact structure and field naming conventions
- Ensure validation compliance with Amazon's requirements
- Apply intelligent business logic for optimal field population

The output must be production-ready for immediate use in Amazon's marketplace system.

**BEGIN TRANSFORMATION NOW**