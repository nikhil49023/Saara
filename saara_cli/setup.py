from setuptools import setup, find_packages

setup(
    name="saara-cli",
    version="1.6.4",
    description="CLI tools for SAARA AI data engine (interactive wizards, commands)",
    author="Kilani Sai Nikhil",
    packages=find_packages(),
    install_requires=[
        "saara-ai>=1.6.4",  # Depends on core package
        "typer[all]",
        "rich>=10.0.0",
        "uvicorn",
    ],
    entry_points={
        "console_scripts": [
            "saara=saara_cli.main:app",
        ],
    },
    python_requires=">=3.8",
    long_description=open("README.md", encoding="utf-8", errors="ignore").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nikhil49023/Data-engine",
    keywords=["llm", "ai", "cli", "interactive", "wizard"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
