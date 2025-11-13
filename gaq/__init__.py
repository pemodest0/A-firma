"""GAQ: Geometric Algebra for Qubits."""

from importlib import metadata

try:
    __version__ = metadata.version("gaq")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["core", "backends", "walks"]
