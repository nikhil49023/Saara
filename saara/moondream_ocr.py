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
        
        # Simpler prompt for Moondream
        self.ocr_prompt = "Read and output the text in this image."

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
                if page_text:
                    all_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
            except Exception as e:
                logger.error(f"Failed to extract page {page_num + 1}: {e}")
                all_text.append(f"--- Page {page_num + 1} ---\n[Extraction failed]")
        
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
