"""Compatibility entry point. Use ``python -m sglang.benchmark.serving``."""

import warnings

from sglang.benchmark.serving import cli_main


if __name__ == "__main__":
    warnings.warn(
        "Use `python -m sglang.benchmark.serving` directly.",
        FutureWarning,
        stacklevel=1,
    )
    cli_main()
