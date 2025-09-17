# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.3,
        thinking_config = types.ThinkingConfig(
            thinking_budget=-1,
        ),
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type = genai.types.Type.OBJECT,
            description = "Structured product data including parent and variant information.",
            required = ["parent_data", "variants"],
            properties = {
                "parent_data": genai.types.Schema(
                    type = genai.types.Type.OBJECT,
                    description = "Information pertaining to the parent product.",
                    required = ["age_range_description", "bottoms_size_class", "bottoms_size_system", "brand_name", "country_of_origin", "department_name", "external_product_id_type", "fabric_type", "feed_product_type", "item_name", "main_image_url", "outer_material_type", "recommended_browse_nodes", "target_gender"],
                    properties = {
                        "age_range_description": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            description = "Description of the target age range for the product.",
                        ),
                        "bottoms_size_class": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            description = "The size classification system for bottoms (e.g., Bundweite & Schrittlänge).",
                        ),
                        "bottoms_size_system": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            description = "The specific size system used for bottoms (e.g., DE / NL / SE / PL).",
                        ),
                        "brand_name": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            description = "The brand name of the product.",
                        ),
                        "country_of_origin": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            description = "The country where the product was manufactured.",
                        ),
                        "department_name": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            description = "The department or general category the product belongs to.",
                        ),
                        "external_product_id_type": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            description = "The type of external product identifier (e.g., EAN, UPC).",
                        ),
                        "fabric_type": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            description = "The primary fabric type of the product.",
                        ),
                        "feed_product_type": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            description = "The product type as categorized for data feeds (e.g., pants, shirt).",
                        ),
                        "item_name": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            description = "The name of the product item.",
                        ),
                        "main_image_url": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            description = "The URL of the main image for the product.",
                            format = "uri",
                        ),
                        "outer_material_type": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            description = "The type of material used for the outer layer of the product.",
                        ),
                        "recommended_browse_nodes": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            description = "A recommended category or browse node ID for the product.",
                        ),
                        "target_gender": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            description = "The target gender for the product (e.g., Männlich, Weiblich, Unisex).",
                        ),
                    },
                ),
                "variants": genai.types.Schema(
                    type = genai.types.Type.ARRAY,
                    description = "A list of product variants, each with its specific attributes.",
                    items = genai.types.Schema(
                        type = genai.types.Type.OBJECT,
                        required = ["color_map", "color_name", "external_product_id", "item_sku", "list_price_with_tax", "quantity", "size_map", "size_name", "standard_price"],
                        properties = {
                            "color_map": genai.types.Schema(
                                type = genai.types.Type.STRING,
                                description = "A standardized or mapped color name for the variant.",
                            ),
                            "color_name": genai.types.Schema(
                                type = genai.types.Type.STRING,
                                description = "The specific color name of the variant.",
                            ),
                            "external_product_id": genai.types.Schema(
                                type = genai.types.Type.STRING,
                                description = "The unique external identifier for this specific variant (e.g., EAN).",
                            ),
                            "item_sku": genai.types.Schema(
                                type = genai.types.Type.STRING,
                                description = "The Stock Keeping Unit (SKU) for this variant.",
                            ),
                            "list_price_with_tax": genai.types.Schema(
                                type = genai.types.Type.STRING,
                                description = "The list price of the variant, including applicable taxes.",
                            ),
                            "quantity": genai.types.Schema(
                                type = genai.types.Type.STRING,
                                description = "The available quantity of this variant.",
                            ),
                            "size_map": genai.types.Schema(
                                type = genai.types.Type.STRING,
                                description = "A standardized or mapped size name for the variant.",
                            ),
                            "size_name": genai.types.Schema(
                                type = genai.types.Type.STRING,
                                description = "The specific size name of the variant.",
                            ),
                            "standard_price": genai.types.Schema(
                                type = genai.types.Type.STRING,
                                description = "The standard selling price of the variant, excluding taxes (if 'list_price_with_tax' includes them).",
                            ),
                        },
                    ),
                ),
            },
        ),
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()
