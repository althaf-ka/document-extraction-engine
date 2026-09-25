class DocumentExtractorError(Exception):
    """Base exception for document extractor errors."""


class InputFileNotFoundError(DocumentExtractorError):
    """Raised when an input file does not exist."""


class UnsupportedFileError(DocumentExtractorError):
    """Raised when an input file format is unsupported."""
