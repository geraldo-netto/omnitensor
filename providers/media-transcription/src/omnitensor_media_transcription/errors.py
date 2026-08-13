"""Provider-specific qualification failures."""


class QualifiedMediaError(RuntimeError):
    """Local provider does not match its frozen qualification."""


__all__ = ["QualifiedMediaError"]
