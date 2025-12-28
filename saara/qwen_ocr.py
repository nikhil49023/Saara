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
        
        # OCR prompt optimized for document extraction
        self.ocr_prompt = """Extract ALL text from this document page. 
Preserve the structure including:
- Headers and titles
- Paragraphs
- Tables (format as markdown tables)
- Lists (preserve numbering/bullets)
- Any Sanskrit/Hindi terms with their transliterations

Output the extracted text exactly as it appears, maintaining layout.
Do NOT summarize or interpret - just extract the raw text."""

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
                if page_text:
                    all_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
            except Exception as e:
                logger.error(f"Failed to extract page {page_num + 1}: {e}")
                all_text.append(f"--- Page {page_num + 1} ---\n[Extraction failed]")
        
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
