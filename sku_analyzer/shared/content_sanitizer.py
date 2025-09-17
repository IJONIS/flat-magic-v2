"""Content sanitization for Gemini AI safety filters.

This module provides German language content sanitization to prevent
safety filter blocking during AI mapping operations.
"""

from typing import Dict, List
import re


class ContentSanitizer:
    """Sanitizes product content to avoid Gemini safety filter triggers."""
    
    GERMAN_TRIGGER_REPLACEMENTS: Dict[str, str] = {
        # Primary triggers identified from parent 4301 analysis
        "Dauerbrenner": "beliebtes Produkt",
        "ältere Herren und Skater": "verschiedene Kundengruppen",
        "lieben die weiten Beine": "bevorzugen den lockeren Schnitt",
        
        # Additional demographic targeting phrases
        "ältere Herren": "erfahrene Kunden",
        "junge Männer": "jüngere Kunden", 
        "Skater": "Freizeitsportler",
        
        # Body-related terminology that could be misinterpreted
        "weiten Beine": "lockeren Schnitt",
        "enge Passform": "figurbetonte Passform",
        
        # Fire/burning related terms
        "Dauerbrenner": "Klassiker",
        "brennend": "dringend",
        "Feuer": "Energie"
    }
    
    AGGRESSIVE_REPLACEMENTS: Dict[str, str] = {
        # More aggressive sanitization for retry attempts
        "Herren": "Personen",
        "Männer": "Personen", 
        "Frauen": "Personen",
        "Damen": "Personen",
        "lieben": "bevorzugen",
        "hassen": "meiden",
        "sexy": "attraktiv",
        "heiß": "beliebt"
    }
    
    def __init__(self, aggressive_mode: bool = False):
        """Initialize sanitizer.
        
        Args:
            aggressive_mode: Use more aggressive sanitization for retry attempts
        """
        self.aggressive_mode = aggressive_mode
        self.replacements = (
            {**self.GERMAN_TRIGGER_REPLACEMENTS, **self.AGGRESSIVE_REPLACEMENTS}
            if aggressive_mode
            else self.GERMAN_TRIGGER_REPLACEMENTS
        )
    
    def sanitize_text(self, text: str) -> str:
        """Sanitize text by replacing trigger words/phrases.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text with trigger words replaced
        """
        if not isinstance(text, str):
            return text
            
        sanitized = text
        
        # Apply replacements (longest phrases first to avoid partial replacements)
        sorted_triggers = sorted(self.replacements.keys(), key=len, reverse=True)
        
        for trigger in sorted_triggers:
            if trigger in sanitized:
                sanitized = sanitized.replace(trigger, self.replacements[trigger])
        
        return sanitized
    
    def sanitize_product_data(self, product_data: Dict) -> Dict:
        """Sanitize all text fields in product data.
        
        Args:
            product_data: Product data dictionary
            
        Returns:
            Sanitized product data dictionary
        """
        if not isinstance(product_data, dict):
            return product_data
            
        sanitized_data = {}
        
        for key, value in product_data.items():
            if isinstance(value, str):
                sanitized_data[key] = self.sanitize_text(value)
            elif isinstance(value, dict):
                sanitized_data[key] = self.sanitize_product_data(value)
            elif isinstance(value, list):
                sanitized_data[key] = [
                    self.sanitize_product_data(item) if isinstance(item, dict)
                    else self.sanitize_text(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                sanitized_data[key] = value
        
        return sanitized_data
    
    def scan_for_triggers(self, text: str) -> List[str]:
        """Scan text for potential trigger words.
        
        Args:
            text: Text to scan
            
        Returns:
            List of found trigger words
        """
        if not isinstance(text, str):
            return []
            
        found_triggers = []
        
        for trigger in self.replacements.keys():
            if trigger in text:
                found_triggers.append(trigger)
        
        return found_triggers
    
    def assess_risk_score(self, product_data: Dict) -> float:
        """Assess content risk score for safety filter blocking.
        
        Args:
            product_data: Product data to assess
            
        Returns:
            Risk score between 0.0 (safe) and 1.0 (high risk)
        """
        if not isinstance(product_data, dict):
            return 0.0
            
        total_triggers = 0
        total_text_length = 0
        
        def count_triggers_recursive(data):
            nonlocal total_triggers, total_text_length
            
            if isinstance(data, str):
                total_text_length += len(data)
                triggers = self.scan_for_triggers(data)
                total_triggers += len(triggers)
            elif isinstance(data, dict):
                for value in data.values():
                    count_triggers_recursive(value)
            elif isinstance(data, list):
                for item in data:
                    count_triggers_recursive(item)
        
        count_triggers_recursive(product_data)
        
        # Calculate risk based on trigger density and absolute count
        if total_text_length == 0:
            return 0.0
            
        trigger_density = total_triggers / max(total_text_length, 1) * 1000  # Per 1000 chars
        
        # Risk calculation: trigger count + density factor
        risk_score = min(1.0, (total_triggers * 0.1) + (trigger_density * 0.05))
        
        return risk_score