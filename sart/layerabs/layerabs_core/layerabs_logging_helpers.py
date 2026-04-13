"""Shared logging helpers for LayerABS family entrypoints and core runners."""

from __future__ import annotations

import os
import sys
import time


class StreamLogger:
    def __init__(self, filename="default.log", stream=None):
        self.terminal = sys.stdout if stream is None else stream
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


class LazyTimestampedLog:
    def __init__(self, module_file, stream=None):
        self.module_file = module_file
        self.stream = stream
        self.style_time = None
        self.is_configured = False

    def ensure_configured(self):
        """Configure stdout mirroring once and return the timestamp string."""
        if self.is_configured:
            return self.style_time

        script_filename = os.path.basename(self.module_file)
        script_name_without_extension = os.path.splitext(script_filename)[0]
        self.style_time = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(time.time()),
        )
        log_path = (
            f"../result/log/{script_name_without_extension}_log_{self.style_time}.txt"
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        sys.stdout = StreamLogger(
            log_path,
            sys.stdout if self.stream is None else self.stream,
        )
        self.is_configured = True
        return self.style_time

    def __str__(self):
        return self.ensure_configured()

    def __repr__(self):
        return repr(str(self))

    def __format__(self, format_spec):
        return format(str(self), format_spec)


def redirect_stdout_to_timestamped_log(module_file, stream=None):
    """Return a lazy timestamped stdout mirror handle for one module."""
    return LazyTimestampedLog(module_file, stream)
