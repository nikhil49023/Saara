"""
Qwen Vision OCR Module
Uses Qwen 2.5 VL via Ollama for powerful document OCR.
"""

import base64
import fitz  # PyMuPDF
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import ollama

logger = logging.getLogger(__name__)


class QwenVisionOCR:
    """
    OCR using Qwen 2.5 Vision Language Model.
    Converts PDF pages to images and extracts text using vision AI.
    """
    
    def __init__(self, config: Dict[str, Any] = None, model_name: str = None):
        self.config = config or {}
        # Allow model override, default to 7b but user might want 2b
        self.model = model_name or self.config.get('model', "qwen2.5vl:7b")
        self.client = ollama.Client(
            host=self.config.get('ollama', {}).get('base_url', 'http://localhost:11434')
        )
        
        # OCR prompt optimized for document extraction - MORE EXPLICIT
        self.ocr_prompt = """You are an OCR system. Your ONLY task is to read and transcribe the text from this document image.

CRITICAL INSTRUCTIONS:
1. Output ONLY the text you see in the image
2. Do NOT describe the image (no "The image shows..." or "This page contains...")
3. Do NOT interpret or summarize - just transcribe exactly
4. Preserve structure: headers, paragraphs, lists, tables
5. If you see Sanskrit/Hindi text, transcribe it with transliteration if possible
6. If a page is blank or unreadable, output: [UNREADABLE PAGE]

START TRANSCRIBING THE TEXT NOW:"""

        # Patterns that indicate bad OCR output (describing instead of extracting)
        self.bad_patterns = [
            "the image shows",
            "this image shows",
            "the page shows",
            "the document shows",
            "the text appears",
            "appears to be",
            "seems to be",
            "this appears to be",
            "the image contains",
            "displays a",
            "difficult to read",
            "foreign language",
            "i can see",
            "i cannot",
            "looking at this",
        ]

    def _is_valid_ocr_output(self, text: str) -> bool:
        """Check if OCR output is actual text extraction, not image description."""
        if not text or len(text.strip()) < 20:
            return False
        
        text_lower = text.lower()[:500]  # Check first 500 chars
        
        # Count how many bad patterns appear
        bad_count = sum(1 for pattern in self.bad_patterns if pattern in text_lower)
        
        # If more than 2 bad patterns, likely describing not extracting
        if bad_count >= 2:
            logger.warning(f"OCR output appears to describe image instead of extracting text (bad patterns: {bad_count})")
            return False
        
        return True

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from entire PDF using Qwen Vision.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Full extracted text from all pages
        """
        logger.info(f"Starting Qwen Vision OCR for: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        all_text = []
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            logger.info(f"Processing page {page_num + 1}/{total_pages}")
            
            try:
                page_text = self._extract_page(doc, page_num)
                
                # Validate the output
                if page_text and self._is_valid_ocr_output(page_text):
                    all_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
                else:
                    # Try with a stricter prompt
                    logger.warning(f"Page {page_num + 1}: Invalid OCR output, trying stricter prompt...")
                    page_text = self._extract_page_strict(doc, page_num)
                    if page_text and self._is_valid_ocr_output(page_text):
                        all_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
                    else:
                        # Fall back to PyMuPDF parser
                        logger.warning(f"Page {page_num + 1}: Using parser fallback")
                        page = doc[page_num]
                        parser_text = page.get_text("text")
                        if parser_text and len(parser_text.strip()) > 50:
                            all_text.append(f"--- Page {page_num + 1} ---\n{parser_text}")
                        
            except Exception as e:
                logger.error(f"Failed to extract page {page_num + 1}: {e}")
                # Try parser fallback
                try:
                    page = doc[page_num]
                    parser_text = page.get_text("text")
                    if parser_text and len(parser_text.strip()) > 50:
                        all_text.append(f"--- Page {page_num + 1} ---\n{parser_text}")
                except:
                    pass
        
        doc.close()
        
        full_text = "\n\n".join(all_text)
        logger.info(f"Qwen Vision OCR complete: {len(full_text)} chars extracted")
        
        return full_text
    
    def _extract_page(self, doc: fitz.Document, page_num: int) -> str:
        """Extract text from a single page using Qwen Vision."""
        
        page = doc[page_num]
        
        # Render page to image (high resolution for OCR)
        # Reduced to 1.5 to save memory/tokens while maintaining readability
        mat = fitz.Matrix(1.5, 1.5)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to base64
        img_bytes = pix.tobytes("png")
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Send to Qwen Vision
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": self.ocr_prompt,
                        "images": [img_base64]
                    }
                ]
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Qwen Vision API error: {e}")
            raise
    
    def _extract_page_strict(self, doc: fitz.Document, page_num: int) -> str:
        """Extract with stricter prompt to avoid image descriptions."""
        
        page = doc[page_num]
        mat = fitz.Matrix(1.5, 1.5)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        strict_prompt = """TRANSCRIBE TEXT ONLY. 
Do not describe. Do not analyze. Just read the words.
Output format: exact text from the image, nothing else.

TEXT:"""
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": strict_prompt,
                        "images": [img_base64]
                    }
                ]
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Strict OCR failed: {e}")
            return None
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from a single image file."""
        
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user", 
                    "content": self.ocr_prompt,
                    "images": [img_base64]
                }
            ]
        )
        
        return response['message']['content']


def test_qwen_ocr():
    """Test the Qwen Vision OCR."""
    ocr = QwenVisionOCR()
    
    # Test with a sample PDF
    # test_pdf = r"E:\Data Set\RAW\AAYUSH DATASET\05112021_Ayurveda_A_Focus_on_Research__Development.pdf"
    test_pdf = r"e:\datapipeline\sample_research_paper.pdf"
    
    if Path(test_pdf).exists():
        text = ocr.extract_text_from_pdf(test_pdf)
        print(f"Extracted {len(text)} characters (7b model)")
        
        # Test 2.5-VL 3B model if available
        print("\nTesting Qwen2.5-VL 3B model...")
        try:
            ocr_2b = QwenVisionOCR(model_name="qwen2.5vl:3b")
            text_2b = ocr_2b.extract_text_from_pdf(test_pdf)
            print(f"Extracted {len(text_2b)} characters (3b model)")
            print(text_2b[:2000])
        except Exception as e:
            print(f"3B model test failed (maybe not pulled yet): {e}")
            
    else:
        print(f"Test PDF not found: {test_pdf}")


if __name__ == "__main__":
    test_qwen_ocr()
