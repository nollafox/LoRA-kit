"""Domain errors raised by lorakit."""


class LorakitError(Exception):
    """Base error; the user-facing message is ``str(error)``."""


class DatasetNotFound(LorakitError):
    """Raised when a dataset does not exist."""


class DatasetExists(LorakitError):
    """Raised when creating or renaming to an existing dataset."""


class CandidateNotFound(LorakitError):
    """Raised when a candidate cannot be resolved."""


class MissingMetadata(LorakitError):
    """Raised when an image cannot resolve required metadata."""


class ModelAmbiguous(LorakitError):
    """Raised when a model name matches multiple model files."""


class ModelNotFound(LorakitError):
    """Raised when a model cannot be resolved."""


class ImporterMissing(LorakitError):
    """Raised when an importer command is unavailable."""


class InvalidPrepareConfig(LorakitError):
    """Raised when prepare options conflict or are unsupported."""
