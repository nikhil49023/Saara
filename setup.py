from setuptools import setup, find_packages

setup(
    name="saara",
    version="0.1.0",
    description="Saara: Autonomous Document-to-LLM Data Factory SDK",
    author="Kilani Sai Nikhil",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.4.0",
        "datasets>=2.10.0",
        "bitsandbytes>=0.40.0",
        "accelerate>=0.20.0",
        "rich>=10.0.0",

        "ollama",
        "fastapi",
        "uvicorn",
        "pydantic",
        "python-multipart",
        "typer[all]"
    ],
    extras_require={
        "dev": ["pytest", "black", "isort", "twine"],
    },
    entry_points={
        "console_scripts": [
            "saara=saara.cli:main",
        ],
    },
    python_requires=">=3.8",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nikhil49023/Data-engine",
    keywords=["llm", "ai", "pdf", "ocr", "fine-tuning", "rag", "dataset", "autonomous"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
