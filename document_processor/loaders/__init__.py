"""Document loaders package."""
from .pdf_loader import PDFLoader
from .document_loader import DOCXLoader
from .factory import LoaderFactory

__all__ = [
    "PDFLoader",
    "DOCXLoader",
    "LoaderFactory",
]
