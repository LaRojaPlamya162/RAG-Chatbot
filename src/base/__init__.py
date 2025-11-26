"""Package initialisation for src.base.

This ensures `src.base` is treated as a package and allows `from src.base ...` imports to work reliably.
"""

__all__ = ["file_loader", "offline_rag", "vector_db", "main"]
