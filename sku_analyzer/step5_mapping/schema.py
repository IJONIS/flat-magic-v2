"""Schema definitions for structured output in AI mapping operations.

This module defines the Gemini API structured output schema that ensures
consistent JSON responses with all required fields properly formatted.
"""

from __future__ import annotations

from typing import Any, Dict
from google import genai
from google.genai import types


def get_ai_mapping_schema() -> types.Schema:
    """Get structured output schema for AI mapping operations.
    
    This schema matches the exact structure from ai_studio_code.py and ensures
    all 23 mandatory fields are properly defined with correct types and descriptions.
    
    Returns:
        Complete Gemini schema for structured output
    """
    return types.Schema(
        type=types.Type.OBJECT,
        description="Structured product data including parent and variant information.",
        required=["parent_data", "variants"],
        properties={
            "parent_data": types.Schema(
                type=types.Type.OBJECT,
                description="Information pertaining to the parent product.",
                required=[
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
                properties={
                    "age_range_description": types.Schema(
                        type=types.Type.STRING,
                        description="Description of the target age range for the product.",
                    ),
                    "bottoms_size_class": types.Schema(
                        type=types.Type.STRING,
                        description="The size classification system for bottoms (e.g., Bundweite & Schrittlänge).",
                    ),
                    "bottoms_size_system": types.Schema(
                        type=types.Type.STRING,
                        description="The specific size system used for bottoms (e.g., DE / NL / SE / PL).",
                    ),
                    "brand_name": types.Schema(
                        type=types.Type.STRING,
                        description="The brand name of the product.",
                    ),
                    "country_of_origin": types.Schema(
                        type=types.Type.STRING,
                        description="The country where the product was manufactured.",
                    ),
                    "department_name": types.Schema(
                        type=types.Type.STRING,
                        description="The department or general category the product belongs to.",
                    ),
                    "external_product_id_type": types.Schema(
                        type=types.Type.STRING,
                        description="The type of external product identifier (e.g., EAN, UPC).",
                    ),
                    "fabric_type": types.Schema(
                        type=types.Type.STRING,
                        description="The primary fabric type of the product.",
                    ),
                    "feed_product_type": types.Schema(
                        type=types.Type.STRING,
                        description="The product type as categorized for data feeds (e.g., pants, shirt).",
                    ),
                    "item_name": types.Schema(
                        type=types.Type.STRING,
                        description="The name of the product item.",
                    ),
                    "main_image_url": types.Schema(
                        type=types.Type.STRING,
                        description="The URL of the main image for the product.",
                        format="uri",
                    ),
                    "outer_material_type": types.Schema(
                        type=types.Type.STRING,
                        description="The type of material used for the outer layer of the product.",
                    ),
                    "recommended_browse_nodes": types.Schema(
                        type=types.Type.STRING,
                        description="A recommended category or browse node ID for the product.",
                    ),
                    "target_gender": types.Schema(
                        type=types.Type.STRING,
                        description="The target gender for the product (e.g., Männlich, Weiblich, Unisex).",
                    ),
                },
            ),
            "variants": types.Schema(
                type=types.Type.ARRAY,
                description="A list of product variants, each with its specific attributes.",
                items=types.Schema(
                    type=types.Type.OBJECT,
                    required=[
                        "color_map",
                        "color_name",
                        "external_product_id",
                        "item_sku",
                        "list_price_with_tax",
                        "quantity",
                        "size_map",
                        "size_name",
                        "standard_price"
                    ],
                    properties={
                        "color_map": types.Schema(
                            type=types.Type.STRING,
                            description="A standardized or mapped color name for the variant.",
                        ),
                        "color_name": types.Schema(
                            type=types.Type.STRING,
                            description="The specific color name of the variant.",
                        ),
                        "external_product_id": types.Schema(
                            type=types.Type.STRING,
                            description="The unique external identifier for this specific variant (e.g., EAN).",
                        ),
                        "item_sku": types.Schema(
                            type=types.Type.STRING,
                            description="The Stock Keeping Unit (SKU) for this variant.",
                        ),
                        "list_price_with_tax": types.Schema(
                            type=types.Type.STRING,
                            description="The list price of the variant, including applicable taxes.",
                        ),
                        "quantity": types.Schema(
                            type=types.Type.STRING,
                            description="The available quantity of this variant.",
                        ),
                        "size_map": types.Schema(
                            type=types.Type.STRING,
                            description="A standardized or mapped size name for the variant.",
                        ),
                        "size_name": types.Schema(
                            type=types.Type.STRING,
                            description="The specific size name of the variant.",
                        ),
                        "standard_price": types.Schema(
                            type=types.Type.STRING,
                            description="The standard selling price of the variant, excluding taxes (if 'list_price_with_tax' includes them).",
                        ),
                    },
                ),
            ),
        },
    )


def get_schema_field_mappings() -> Dict[str, str]:
    """Get mapping between old format fields and structured output fields.
    
    Returns:
        Dictionary mapping old field names to new structured output field names
    """
    return {
        # Parent data mappings
        "brand_name": "parent_data.brand_name",
        "feed_product_type": "parent_data.feed_product_type", 
        "item_name": "parent_data.item_name",
        "department_name": "parent_data.department_name",
        "fabric_type": "parent_data.fabric_type",
        "country_of_origin": "parent_data.country_of_origin",
        "target_gender": "parent_data.target_gender",
        "age_range_description": "parent_data.age_range_description",
        "bottoms_size_class": "parent_data.bottoms_size_class",
        "bottoms_size_system": "parent_data.bottoms_size_system",
        "external_product_id_type": "parent_data.external_product_id_type",
        "main_image_url": "parent_data.main_image_url",
        "outer_material_type": "parent_data.outer_material_type",
        "recommended_browse_nodes": "parent_data.recommended_browse_nodes",
        
        # Variant data mappings
        "item_sku": "variants.item_sku",
        "color_name": "variants.color_name", 
        "color_map": "variants.color_map",
        "size_name": "variants.size_name",
        "size_map": "variants.size_map",
        "external_product_id": "variants.external_product_id",
        "quantity": "variants.quantity",
        "list_price_with_tax": "variants.list_price_with_tax",
        "standard_price": "variants.standard_price",
    }