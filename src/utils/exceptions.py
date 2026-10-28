"""Application-specific exceptions with user-safe messages."""


class MuleShieldError(Exception):
    """Base error. The message is safe to show to an ordinary user."""


class DataValidationError(MuleShieldError):
    """Input data is malformed, too large or does not match the schema."""


class AnalysisError(MuleShieldError):
    """The analysis pipeline could not complete."""


class CaseError(MuleShieldError):
    """Invalid investigation action (e.g. resolving without a reason)."""
