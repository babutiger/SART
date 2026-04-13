"""Shared compatibility helpers for legacy LayerABS aliases."""

from __future__ import annotations

__all__ = [
    "resolve_legacy_alias",
    "resolve_legacy_module_attribute",
]


def resolve_legacy_alias(module_name, name, aliases):
    """Resolve one legacy alias or raise a standard module-style error."""
    try:
        return aliases[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {module_name!r} has no attribute {name!r}"
        ) from exc


def resolve_legacy_module_attribute(module_name, name, load_module):
    """Resolve one attribute through a lazily loaded legacy implementation."""
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(
            f"module {module_name!r} has no attribute {name!r}"
        )
    try:
        return getattr(load_module(), name)
    except AttributeError as exc:
        raise AttributeError(
            f"module {module_name!r} has no attribute {name!r}"
        ) from exc
