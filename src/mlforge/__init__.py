"""Saara local-first ML workflow CLI."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("saara-ai")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "1.8.0"
