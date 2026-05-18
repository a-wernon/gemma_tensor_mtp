from .base import MTPCHead
from .ff import FFClusterHead
from .cp import CPClusterHead

__all__ = ["MTPCHead", "FFClusterHead", "CPClusterHead"]


def build_head(name: str, **kwargs) -> MTPCHead:
    if name == "ff":
        return FFClusterHead(**kwargs)
    if name == "cp":
        return CPClusterHead(**kwargs)
    raise ValueError(f"Unknown head: {name!r}")
