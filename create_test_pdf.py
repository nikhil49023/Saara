
import fitz

def create_sample_pdf(filename="sample_research_paper.pdf"):
    doc = fitz.open()
    page = doc.new_page()
    
    # Title
    page.insert_text((50, 50), "Advances in Automated Data Pipelines", fontsize=18)
    
    # Author
    page.insert_text((50, 80), "By Antigravity Agent", fontsize=12)
    
    # Abstract
    page.insert_text((50, 120), "Abstract", fontsize=14)
    abstract_text = """
    This paper explores the development of automated data pipelines for training Large Language Models (LLMs).
    We discuss the integration of PDF extraction, semantic text chunking, and intelligent labeling using
    local LLMs like Granite 4.0. The proposed system demonstrates significant improvements in data preparation efficiency.
    """
    page.insert_textbox(fitz.Rect(50, 140, 500, 250), abstract_text, fontsize=10)
    
    # Introduction
    page.insert_text((50, 270), "1. Introduction", fontsize=14)
    intro_text = """
    The quality of training data is paramount for the performance of machine learning models.
    Traditional manual labeling is time-consuming and expensive. Our approach leverages the
    reasoning capabilities of modern SLMs to automate this process.
    
    Key components include:
    * Robust PDF parsing
    * Context-aware segmentation
    * Multi-task labeling (QA, Summarization, classification)
    """
    page.insert_textbox(fitz.Rect(50, 290, 500, 500), intro_text, fontsize=10)
    
    # Save
    doc.save(filename)
    print(f"Created {filename}")

if __name__ == "__main__":
    create_sample_pdf()
