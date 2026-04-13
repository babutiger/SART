"""Shared lazy process-management helpers for LayerABS runners."""

from __future__ import annotations

import multiprocessing

_PROCESS_REGISTRY = {}


def get_managed_child_process_list(registry_key):
    """Return a lazily created shared child-process list for one module key."""
    registry_entry = _PROCESS_REGISTRY.get(registry_key)
    if registry_entry is None:
        process_manager = multiprocessing.Manager()
        child_process_ids = process_manager.list()
        registry_entry = (process_manager, child_process_ids)
        _PROCESS_REGISTRY[registry_key] = registry_entry
    return registry_entry[1]
