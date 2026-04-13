"""Context helpers for temporary LayerABS solver overrides."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

_MIP_TIME_LIMIT_ENV = "LAYERABS_MIP_TIME_LIMIT_SECONDS"


def get_solver_time_limit_seconds() -> float | None:
    raw_value = os.environ.get(_MIP_TIME_LIMIT_ENV)
    if raw_value in {None, ""}:
        return None
    return float(raw_value)


@contextmanager
def override_solver_time_limit_seconds(
    time_limit_seconds: float | None,
) -> Iterator[None]:
    previous_value = os.environ.get(_MIP_TIME_LIMIT_ENV)
    try:
        if time_limit_seconds is None:
            os.environ.pop(_MIP_TIME_LIMIT_ENV, None)
        else:
            os.environ[_MIP_TIME_LIMIT_ENV] = str(time_limit_seconds)
        yield
    finally:
        if previous_value is None:
            os.environ.pop(_MIP_TIME_LIMIT_ENV, None)
        else:
            os.environ[_MIP_TIME_LIMIT_ENV] = previous_value
