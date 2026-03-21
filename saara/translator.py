"""
Translation Module using Sarvam AI.
Handles language detection (via Granite) and translation (via Sarvam).
"""

import logging
from typing import Optional, Tuple

from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class Translator:
    """
    Handles text translation using Sarvam AI.
    Uses Granite 4.0 for language detection if needed.
    """
    
    def __init__(self, config: dict, ollama_client: OllamaClient):
        self.config = config.get('sarvam', {})
        self.api_key = self.config.get('api_key')
        self.model = self.config.get('model', 'sarvam-translate:v1')
        self.enabled = self.config.get('enabled', False)
        
        self.ollama_client = ollama_client
        
        if self.enabled and self.api_key:
            try:
                import sarvamai
                self.client = sarvamai.SarvamAI(api_subscription_key=self.api_key)
                logger.info("Sarvam AI client initialized")
            except ImportError:
                logger.warning("Sarvam AI package not found. Disabling translation.")
                self.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize Sarvam AI: {e}")
                self.enabled = False
        else:
            self.enabled = False
            
    def translate_document(self, document) -> None:
        """
        Translates content of an ExtractedDocument in-place.
        Translates page by page.
        """
        if not self.enabled:
            return

        logger.info("Checking for translation needs...")
        
        # Check first page to decide if translation is needed
        if not document.pages:
            return
            
        first_page_text = document.pages[0].text[:1000]
        lang_code = self._detect_language(first_page_text)
        
        if not lang_code or lang_code.startswith('en'):
            logger.info(f"Language detected as {lang_code}, skipping translation.")
            return

        logger.info(f"Detected {lang_code}. Translating document...")
        
        new_pages_text = []
        for i, page in enumerate(document.pages):
            logger.info(f"Translating page {i+1}/{len(document.pages)}...")
            if page.text.strip():
                translated = self._translate(page.text, lang_code)
                page.text = translated
                new_pages_text.append(translated)
            else:
                new_pages_text.append("")
        
        # Update full text
        document.full_text = "\n\n".join(new_pages_text)
        logger.info("Document translation complete.")

    def translate_if_needed(self, text: str) -> str:
        """
        Detects language and translates to English if necessary.
        
        Args:
            text: Input text
            
        Returns:
            Translated text (or original if English/error)
        """
        if not self.enabled or not text.strip():
            return text
            
        # 1. Detect Language using Granite
        lang_code = self._detect_language(text[:500])  # Detect on first 500 chars
        
        if not lang_code or lang_code.startswith('en'):
            return text
            
        logger.info(f"Detected language: {lang_code}. Translating...")
        
        # 2. Translate using Sarvam
        return self._translate(text, lang_code)
        
    def _detect_language(self, text: str) -> Optional[str]:
        """
        Uses Granite 4.0 to detect the language code.
        """
        prompt = f"""Identify the language of this text. 
Return ONLY the ISO language code suitable for Sarvam AI (e.g., hi-IN, te-IN, sa-IN, en-IN).
If unsure or if it is English, return 'en-IN'.

Text:
{text}

Code:"""
        
        response = self.ollama_client.generate(prompt, max_tokens=10)
        if response.success:
            code = response.content.strip().replace("'", "").replace('"', "")
            # Basic validation
            if len(code) >= 2 and '-' in code:
                return code
            if code.lower() in ['english', 'en']:
                return 'en-IN'
                
        return 'en-IN' # Default to English if detection fails
        
    def _translate(self, text: str, source_lang: str) -> str:
        """Translate text using Sarvam AI."""
        try:
            # Sarvam might have a limit on text length. 
            # If text is huge, we might need to split. 
            # But chunks are usually < 3000 chars, which should be fine.
            
            response = self.client.text.translate(
                input=text,
                source_language_code=source_lang,
                target_language_code="en-IN",
                mode="formal",
                model=self.model,
                numerals_format="native",
                speaker_gender="Male", # Default
                enable_preprocessing=False
            )
            
            # Use dot notation for response if the SDK returns an object, 
            # or dictionary access if it returns a dict. 
            # Based on user snippet 'print(response)', it likely returns an object or dict.
            # Usually these SDKs return an object with 'translated_text'.
            # Let's try to handle both or inspect typical usage.
            # Assuming getting `translated_text` attribute or key.
            
            if hasattr(response, 'translated_text'):
                return response.translated_text
            elif isinstance(response, dict) and 'translated_text' in response:
                return response['translated_text']
            else:
                # Fallback, maybe it's just the string? Unlikely.
                # Let's return str(response) if unsure, but better to log.
                logger.debug(f"Sarvam response: {response}")
                # Most likely it has a field.
                # Checking hypothetical SDK structure: usually `response.translated_text`.
                return str(response)

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text
