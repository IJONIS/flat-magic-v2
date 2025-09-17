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
  "metadata": {
    "generation_timestamp": "2025-09-04T23:39:45.242000",
    "source_file": "step3_mandatory_fields.json",
    "template_version": "1.0",
    "field_distribution": {
      "parent_field_count": 13,
      "variant_field_count": 10,
      "total_field_count": 23,
      "parent_ratio": 0.5652173913043478,
      "variant_ratio": 0.43478260869565216
    },
    "quality_score": 1.0,
    "validation_status": "valid",
    "warnings": [],
    "categorization_method": "ai",
    "ai_confidence": 0.98
  },
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
              "TMG",
              "TMG INTERNATIONAL Textile Management Group",
              "TMG Clothing"
            ]
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
              "GTIN",
              "UPC",
              "ASIN",
              "EAN",
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
        "outer_material_type": {
          "display_name": "Hauptmaterial",
          "data_type": "string",
          "constraints": {
            "value_count": 33,
            "max_length": 17
          },
          "applies_to_children": false,
          "validation_rules": {
            "required": false,
            "data_type": "string",
            "max_length": 17,
            "allowed_values": [
              "Wolle",
              "Synthetik",
              "Satin",
              "Kashmir",
              "Hanf",
              "Fur",
              "Pelz",
              "Fleece",
              "Glattleder",
              "Lackleder",
              "Gummi",
              "Jeans",
              "Wildleder",
              "Kaschmir",
              "Synthetisch",
              "Angora",
              "Pailletten",
              "Leinen",
              "Pelzimitat",
              "Mohair",
              "Daunen",
              "Leder",
              "Samt",
              "Alpaka",
              "Denim",
              "Cord",
              "Seide",
              "Angorawolle",
              "Fellimitat",
              "Paillettenbesetzt",
              "Filz",
              "Merino",
              "Baumwolle"
            ]
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
              "Weiblich",
              "Unisex",
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
              "Erwachsener",
              "Kleinkind",
              "Baby",
              "Kind"
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
          "applies_to_children": true,
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
          "applies_to_children": true,
          "validation_rules": {
            "required": true,
            "data_type": "string",
            "max_length": 24,
            "allowed_values": [
              "Alter",
              "Alphanumerisch",
              "Numerisch",
              "Bundweite",
              "Bundweite & Schrittlänge"
            ]
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
              "Unisex Kinder",
              "Mädchen",
              "Jungen",
              "Unisex",
              "Damen",
              "Unisex Baby",
              "Baby - Jungen",
              "Baby - Mädchen",
              "Herren"
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
          "applies_to_children": false,
          "validation_rules": {
            "required": false,
            "data_type": "string",
            "max_length": 33,
            "allowed_values": [
              "Australien",
              "Bolivien",
              "Brasilien",
              "Französisch-Polynesien",
              "Griechenland",
              "Honduras",
              "Kiribati",
              "Neukaledonien",
              "Turks-und Caicosinseln",
              "Kambodscha",
              "Liechtenstein",
              "Philippinen",
              "Niger",
              "Antarktika",
              "Guernsey",
              "Somalia",
              "Österreich",
              "Eritrea",
              "Mauritius",
              "Guam",
              "Nordkorea",
              "Swasiland",
              "Timor-Leste",
              "Mongolei",
              "Togo",
              "Komoren",
              "XY",
              "Tschechische Republik",
              "Algerien",
              "Bulgarien",
              "Kamerun",
              "Nepal",
              "Spanien",
              "Saudi-Arabien",
              "Polen",
              "Puerto Rico",
              "Iran",
              "Schweden",
              "Kasachstan",
              "Dänemark",
              "Taiwan",
              "Lesotho",
              "Usbekistan",
              "Syrien",
              "Pitcairn-Inseln",
              "British Virgin Islands",
              "Bhutan",
              "Kanarische Inseln",
              "Palau",
              "Chile",
              "Sambia",
              "Mosambik",
              "Kuba",
              "Salomon-Inseln",
              "Nauru",
              "Saint Vincent und die Grenadinen",
              "Uruguay",
              "Äquatorial-Guinea",
              "Demokratische Republik Kongo",
              "St. Martin (französischer Teil)",
              "XM",
              "Malta",
              "Tonga",
              "Libyen",
              "Israel",
              "Georgien",
              "US-Ozeanien",
              "Guyana",
              "Libanon",
              "Costa Rica",
              "Kuwait",
              "Mexiko",
              "Tunesien",
              "Haiti",
              "Ägypten",
              "Nordmazedonien",
              "Malediven",
              "Bangladesch",
              "Tansania",
              "Falkland-Inseln",
              "Luxemburg",
              "Ruanda",
              "Mayotte",
              "Argentinien",
              "Burma (Myanmar)",
              "Suriname",
              "Bahrein",
              "Litauen",
              "Grönland",
              "Neuseeland",
              "Niederländische Antillen",
              "Südkorea",
              "Elfenbeinküste",
              "Bosnien und Herzegowina",
              "Gibraltar",
              "Slowenien",
              "Heiliger Stuhl (Vatikanstadt)",
              "Réunion",
              "Südafrika",
              "Laos",
              "Finnland",
              "Brunei",
              "Frankreich",
              "Gambia",
              "Irak",
              "Serbien",
              "St. Barthélemy",
              "XK",
              "Sudan",
              "Panama",
              "Indien",
              "Angola",
              "Cocos (Keeling) Inseln",
              "Bonaire, St. Eustatius und Saba",
              "Ecuador",
              "Färöer-Inseln",
              "Botswana",
              "Ghana",
              "Kenia",
              "Macau",
              "Italien",
              "Japan",
              "Norfolk Island",
              "Aruba",
              "Ascension",
              "Sierra Leone",
              "Venezuela",
              "St. Kitts und Nevis",
              "Portugal",
              "Samoa",
              "Seychellen",
              "Malawi",
              "Palästinensische Autonomiegebiete",
              "Marokko",
              "Marshall-Inseln",
              "Großbritannien",
              "Island",
              "Dominica",
              "Christmas Island",
              "Jugoslawien",
              "Kanada",
              "US Virgin Islands",
              "XB",
              "Armenien",
              "Bahamas",
              "Martinique",
              "Simbabwe",
              "XE",
              "Montenegro",
              "Trinidad und Tobago",
              "Papua-Neuguinea",
              "Moldawien",
              "Jamaika",
              "Turkmenistan",
              "Wallis und Futuna",
              "Katar",
              "El Salvador",
              "Gabun",
              "Bouvet-Insel",
              "Burkina Faso",
              "Tuvalu",
              "China",
              "Namibia",
              "Westsahara",
              "Paraguay",
              "Senegal",
              "Singapur",
              "Andorra",
              "Antigua und Barbuda",
              "Albanien",
              "San Marino",
              "Guatemala",
              "Norwegen",
              "Lettland",
              "WD",
              "Nigeria",
              "Peru",
              "Estland",
              "Indonesien",
              "Mikronesien",
              "Kolumbien",
              "Zaire",
              "Rumänien",
              "Cayman Islands",
              "Fidschi",
              "Benin",
              "Niederlande",
              "Aserbaidschan",
              "WZ",
              "Schweiz",
              "Svalbard",
              "São Tomé und Príncipe",
              "Saint Lucia",
              "Vereinigte Staaten",
              "Osttimor",
              "Jordanien",
              "Guadeloupe",
              "Sri Lanka",
              "Hongkong",
              "St. Pierre und Miquelon",
              "Dominikanische Republik",
              "XC",
              "Vanuatu",
              "Cook-Inseln",
              "Grenada",
              "Bermuda",
              "Belize",
              "Republik Kongo",
              "Tadschikistan",
              "Isle of Man",
              "Weißrussland",
              "XN",
              "Oman",
              "Ukraine",
              "Barbados",
              "Irland",
              "Mauretanien",
              "Thailand",
              "Äthiopien",
              "Guinea-Bissau",
              "Niue",
              "Vereinigtes Königreich",
              "Vereinigte Arabische Emirate",
              "S. Georgia und S. Sandwich ISLs.",
              "Kap Verde",
              "Zentralafrikanische Republik",
              "Monaco",
              "Kroatien",
              "Curaçao",
              "Tschad",
              "Montserrat",
              "Liberia",
              "Pakistan",
              "Anguilla",
              "Tokelau",
              "Madagaskar",
              "Slowakei",
              "Northern Mariana Islands",
              "Zypern",
              "Dschibuti",
              "Unbekannt",
              "Mali",
              "Aland-Inseln",
              "Deutschland",
              "Französisch Südliche Territorien",
              "Jersey",
              "Vietnam",
              "Türkei",
              "Kirgisistan",
              "Tristan da Cunha",
              "British Indian Ocean Territory",
              "Südsudan",
              "Französisch-Guayana",
              "Ungarn",
              "Guinea",
              "Belgien",
              "Nicaragua",
              "Afghanistan",
              "Saint Helena",
              "Malaysia",
              "Russland",
              "Heard und McDonald Inseln",
              "Burundi",
              "Saint-Martin",
              "Serbien und Montenegro",
              "Uganda",
              "Amerikanisch-Samoa",
              "Jemen"
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
        }
      },
      "field_count": 13,
      "required_fields": [
        "feed_product_type",
        "brand_name",
        "external_product_id_type",
        "recommended_browse_nodes",
        "target_gender",
        "age_range_description",
        "bottoms_size_system",
        "bottoms_size_class",
        "department_name"
      ]
    },
    "child_variants": {
      "fields": {
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
        },
        "external_product_id": {
          "display_name": "Hersteller-Barcode",
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
        },
        "standard_price": {
          "display_name": "Preis",
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
        },
        "quantity": {
          "display_name": "Anzahl",
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
        },
        "main_image_url": {
          "display_name": "URL des Hauptbilds",
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
              "Rosa",
              "Braun",
              "Türkis",
              "Orange",
              "Rot",
              "Lila",
              "Elfenbein",
              "Gelb",
              "Silber",
              "Bronze",
              "Metallisch",
              "Durchsichtig",
              "Weiß",
              "Grau",
              "Cremefarben",
              "Gold",
              "Mehrfarbig",
              "Beige",
              "Schwarz",
              "Blau",
              "Violett",
              "Grün"
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
              "43",
              "46",
              "26",
              "37",
              "38",
              "One Size",
              "58",
              "56",
              "23",
              "45",
              "31",
              "68",
              "54",
              "32",
              "66",
              "42",
              "48",
              "29",
              "50",
              "40",
              "22",
              "70",
              "36",
              "30",
              "34",
              "25",
              "39",
              "28",
              "47",
              "33",
              "60",
              "35",
              "24",
              "44",
              "52",
              "27",
              "64",
              "62",
              "49",
              "41"
            ]
          }
        },
        "list_price_with_tax": {
          "display_name": "Preis mit Steuern zur Anzeige",
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
      "field_count": 10,
      "variable_fields": [
        "item_sku",
        "external_product_id",
        "standard_price",
        "quantity",
        "main_image_url",
        "color_map",
        "color_name",
        "size_name",
        "size_map",
        "list_price_with_tax"
      ],
      "inherited_fields": [
        "external_product_id",
        "standard_price",
        "quantity",
        "main_image_url",
        "color_map",
        "color_name",
        "size_name",
        "size_map",
        "list_price_with_tax"
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
        },
        {
          "field": "bottoms_size_system",
          "inheritance_type": "mandatory",
          "override_allowed": false
        },
        {
          "field": "bottoms_size_class",
          "inheritance_type": "mandatory",
          "override_allowed": false
        }
      ],
      "variant_overrides": [
        {
          "field": "external_product_id",
          "default_source": "parent",
          "variation_required": true
        },
        {
          "field": "standard_price",
          "default_source": "parent",
          "variation_required": true
        },
        {
          "field": "quantity",
          "default_source": "parent",
          "variation_required": true
        },
        {
          "field": "main_image_url",
          "default_source": "parent",
          "variation_required": true
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
          "field": "list_price_with_tax",
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
    "FNAME_3_25": "Gesäßnaht",
    "FVALUE_3_25": "Die Gesäßnaht wird nach der Methode einer Anzughose gefertigt, was eine individuelle Anpassung der Hose ermöglicht.",
    "FNAME_3_20": "Knietaschen für Kniepolster",
    "MIME_DESCR_3": "Logo",
    "FVALUE_3_12": "zwei eingesetzte Schubtaschen",
    "MANUFACTURER_TYPE_DESCRIPTION": "PERCY",
    "FVALUE_3_15": "extra breite Leistentasche",
    "FNAME_3_17": "Keil im Schritt",
    "MIME_PURPOSE_2": "URL",
    "FNAME_3_24": "Schrittnaht",
    "FVALUE_3_11": "schwarz",
    "FNAME_2_2": "-",
    "MIME_THUMB_2": "https://blob.redpim.de/company-53e006db-2b74-4ce1-5a4d-08dca19c0e21/mimes/_normal.jpg",
    "FNAME_3_1": "Farbcode",
    "FNAME_3_8": "Bundband",
    "FNAME_3_22": "Einschub Knietaschen",
    "ORDER_UNIT": "C62",
    "FNAME_3_10": "Reißverschluss Zähne Farbe",
    "MIME_THUMB_3": "https://blob.redpim.de/company-53e006db-2b74-4ce1-5a4d-08dca19c0e21/mimes/326486_404DFA136769374B5D54953119A6A2E3380034E6ED0EFB67D648AC04236DAB49_normal.jpg",
    "FVALUE_3_17": "Flache Nähte für reibungslosen Komfort und erhöhte Stabilität",
    "DESCRIPTION_LONG": "Diese Zunfthose mit normaler Fußweite sorgt für ein meisterliches Auftreten auf dem Bau oder bei der Repräsentation. Echtlederpaspel und Echtlederecken geben der Dreidrahtcord-Hose eine besonders edle Optik. Durch die klassische Herrenkonfektionierung lässt sich die Hose an der Gesäßnaht in der Größe variieren.",
    "MIME_PURPOSE_3": "Logo",
    "FNAME_3_15": "Seitentasche oben rechts",
    "FVALUE_3_3": "Schwarz",
    "FVALUE_3_18": "6 Knöpfe um Hosenträger mit Patten oder Biesen zu befestigen",
    "FNAME_3_21": "Material Knietaschen",
    "NOCUPEROU": 1,
    "MIME_DESCR_2": "Link",
    "FNAME_3_3": "Farbe",
    "FVALUE_3_7": "100% Polyester extra stark & schwer",
    "FNAME_3_6": "Oberstoff",
    "FVALUE_3_10": "Messing/Gold",
    "MASTER": 41282,
    "INTERVAL_QUANTITY": 1,
    "FVALUE_3_22": "von unten - mit Klettverschluss",
    "MIME_THUMB_1": "https://blob.redpim.de/company-53e006db-2b74-4ce1-5a4d-08dca19c0e21/mimes/4160633_6A261AB71579891EE1DFFB78F85DE71405A04C3B7A6038B2C33C4D5B4B640F52_normal.jpg",
    "CONTENT_UNIT": "C62",
    "FVALUE_3_5": "ZUNFT EXCLUSIV",
    "FVALUE_3_16": "Cordura",
    "FNAME_3_16": "Taschenpaspelierung",
    "FNAME_3_2": "Größe",
    "SYSTEMNAME_1": "udf_NMMARKETINGCLAIM-1.0",
    "FVALUE_3_8": "100% Baumwolle mit EIKO Logo",
    "FNAME_3_9": "Reißverschlussbreite",
    "CUSTOMS_TARIFF_NUMBER": 62034211,
    "FNAME_2_1": "-",
    "GROUP_STRING": "Root|Zunftbekleidung|Zunfthosen",
    "SYSTEMNAME_2": "udf_NMTOPFEATURES-1.0",
    "FVALUE_3_9": "9 mm",
    "WEIGHT": "1,14",
    "FNAME_3_14": "Seitentasche oben links",
    "FVALUE_2_2": "mit Echtlederbesatz",
    "FNAME_3_23": "Ausführung Knietasche",
    "FVALUE_2_1": "Zunfthose",
    "MIME_PURPOSE_1": "Normal",
    "FVALUE_3_19": "rechteckige Verstärkungen aus echtem Vollrindleder",
    "FNAME_3_13": "Gesäßtaschen",
    "MANUFACTURER_NAME": "EIKO",
    "FVALUE_3_21": "Oberstoff",
    "FNAME_3_4": "Fußweite",
    "PRODUCT_STATUS": "ACTIVE",
    "FNAME_3_18": "Knöpfe für Hosenträger",
    "FNAME_3_11": "Reißverschluss Gewebe Farbe",
    "FNAME_3_5": "Serie",
    "FVALUE_3_20": "ja",
    "COUNTRY_OF_ORIGIN": "Tunesien",
    "FNAME_3_7": "Taschenfutter",
    "MIME_DESCR_1": "Hauptbild",
    "SYSTEMNAME_3": "udf_NMTECHNICALDETAILS-1.0",
    "FNAME_3_12": "Schubtaschen vorn",
    "FVALUE_3_1": 40,
    "FVALUE_3_6": "Goliath-Cord",
    "FVALUE_3_23": "von außen erreichbar",
    "FUNIT_3_4": "cm",
    "FVALUE_3_13": "eine Gesäßtasche rechts mit Lasche und Knopfverschluss",
    "MIN_QUANTITY": 1,
    "FVALUE_3_14": "extra breite Leistentasche",
    "MANUFACTURER_PID": 41282,
    "FNAME_3_19": "Taschenverstärkung",
    "FVALUE_3_24": "Eine dreifache Kappnaht bietet erhöhte Festigkeit und Haltbarkeit, ideal für stark beanspruchte Bereiche. Sie ist elastischer als eine normale Naht, da sich der Faden bei Belastung längen kann. Zudem sorgt sie für ein sauberes und flaches Finish.",
    "_parent_sku": "41282",
    "_child_count": 28,
    "_analysis_timestamp": 1757021900.146003
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

## 🧠 COMPREHENSIVE MAPPING INSTRUCTIONS

### **Your Mission**
Transform the authentic German workwear product data into a complete Amazon marketplace JSON format that:
1. Maps ALL product variants
2. Populates all mandatory fields using real source data
3. Uses the template structure and mandatory fields to determine the correct output format
4. Complies with all validation rules from step4_template.json
5. Applies intelligent business logic for field derivation

---

## EXECUTION COMMAND

**GENERATE THE COMPLETE AMAZON MARKETPLACE JSON NOW**

Using the two complete data files provided above:
1. The step4_template.json (validation rules and allowed values)
2. The step2_compressed.json (real product source data)

Create a comprehensive JSON transformation that:
- Processes ALL variants without any truncation
- Map all mandatory fields using authentic source data
- Use the template structure to determine correct field organization (parent vs variant data)
- Ensure validation compliance with Amazon's requirements
- Apply intelligent business logic for optimal field population

The output must be production-ready for immediate use in Amazon's marketplace system.

**BEGIN TRANSFORMATION NOW**