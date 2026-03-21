"""
Moondream OCR Module
Uses Moondream via Ollama for lightweight document OCR.
"""

import base64
import fitz  # PyMuPDF
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MoondreamOCR:
    """
    OCR using Moondream Vision Language Model.
    Converts PDF pages to images and extracts text using vision AI.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = "moondream"
        self.client = ollama.Client(
            host=self.config.get('ollama', {}).get('base_url', 'http://localhost:11434')
        )
        
        # Improved prompt for Moondream - more explicit to avoid descriptions
        self.ocr_prompt = """TRANSCRIBE the text from this document. 
Output ONLY the text content - do not describe the image.
START:"""

        # Patterns that indicate bad OCR output (describing instead of extracting)
        self.bad_patterns = [
            "the image shows",
            "this image shows",
            "the page shows",
            "the document shows",
            "the text appears",
            "appears to be",
            "seems to be",
            "the image contains",
            "difficult to read",
            "foreign language",
            "i can see",
            "looking at this",
        ]

    def _is_valid_ocr_output(self, text: str) -> bool:
        """Check if OCR output is actual text extraction, not image description."""
        if not text or len(text.strip()) < 20:
            return False
        
        text_lower = text.lower()[:500]
        bad_count = sum(1 for pattern in self.bad_patterns if pattern in text_lower)
        
        if bad_count >= 2:
            logger.warning(f"Moondream output appears to describe image (bad patterns: {bad_count})")
            return False
        
        return True

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from entire PDF using Moondream.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Full extracted text from all pages
        """
        logger.info(f"Starting Moondream OCR for: {pdf_path}")
        
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
                    # Fall back to PyMuPDF parser for this page
                    logger.warning(f"Page {page_num + 1}: Moondream gave invalid output, using parser")
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
        logger.info(f"Moondream OCR complete: {len(full_text)} chars extracted")
        
        return full_text
    
    def _extract_page(self, doc: fitz.Document, page_num: int) -> str:
        """Extract text from a single page using Moondream."""
        
        page = doc[page_num]
        
        # Render page to image
        # Moondream is small, so we don't need super high res, but text needs to be clear.
        # Using 1.5 to balance quality and token count/memory
        mat = fitz.Matrix(1.5, 1.5)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to base64
        img_bytes = pix.tobytes("png")
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Send to Moondream
        try:
            start_time = time.time()
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
            duration = time.time() - start_time
            logger.debug(f"Page {page_num + 1} took {duration:.2f}s")
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Moondream API error: {e}")
            raise

def test_moondream_ocr():
    """Test the Moondream OCR."""
    ocr = MoondreamOCR()
    
    # Test with sample PDF
    test_pdf = r"e:\datapipeline\sample_research_paper.pdf"
    
    if Path(test_pdf).exists():
        print(f"Testing with {test_pdf}")
        text = ocr.extract_text_from_pdf(test_pdf)
        print(f"\nExtracted {len(text)} characters")
        print("-" * 50)
        print(text[:2000] if text else "No text extracted")
        print("-" * 50)
    else:
        print(f"Test PDF not found: {test_pdf}")

if __name__ == "__main__":
    test_moondream_ocr()
