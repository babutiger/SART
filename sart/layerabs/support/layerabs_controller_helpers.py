from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Any


def run_variant_controller(
    *,
    description: str,
    default_variant: str,
    run_configured_variant: Callable[..., Any],
    argv: Sequence[str] | None = None,
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
    extra_kwargs_builder: Callable[[argparse.Namespace], dict[str, Any]] | None = None,
) -> None:
    """Parse a model-neutral family CLI and forward to the configured runner."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--variant",
        default=default_variant,
        help=f"Variant key from the family's variant table. Default: {default_variant}.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Override the variant's default perturbation radius.",
    )
    if add_arguments is not None:
        add_arguments(parser)
    args = parser.parse_args(argv)

    extra_kwargs = (
        extra_kwargs_builder(args) if extra_kwargs_builder is not None else {}
    )
    run_configured_variant(args.variant, d=args.delta, **extra_kwargs)
