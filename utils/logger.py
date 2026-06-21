"""
Thread-safe logger that writes to both stdout and an optional log file.

Usage
-----
    from utils.logger import log
    log.set_log_file("output/train.log")
    log.print("Training started")
    log.info("Epoch 1 loss = 0.42")
"""

import os
import sys
import threading
from datetime import datetime
from typing import Optional


class _Logger:
    """
    Thread-safe logger with dual output: stdout + optional file.

    Level-based methods (``info``, ``warning``, ``error``) prepend a
    timestamp and level tag.  The raw ``print`` method writes messages
    as-is (no decoration).

    Public API
    ----------
    set_log_file(path)
        Enable file logging.  Creating the parent directory is attempted
        automatically.
    print(*args, sep, end, flush)
        Write raw message.
    info / warning / error(*args, sep, end, flush)
        Write timestamped, level-tagged message.
    """

    def __init__(self):
        self._log_file: Optional[str] = None
        self._lock = threading.Lock()
        self._file_enabled = True
        self._warned = False

    def set_log_file(self, log_file: str):
        """
        Enable file logging to *log_file*.

        Creates the parent directory if needed.  If the file cannot be
        opened, file logging is disabled and a warning is printed to stderr.
        """
        self._log_file = log_file
        self._file_enabled = True
        self._warned = False

        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(self._log_file, "a", encoding="utf-8"):
                pass
        except Exception as e:
            self._file_enabled = False
            sys.stderr.write(
                f"[Logger WARNING] Failed to open log file '{self._log_file}': {e}\n"
            )

    def print(self, *args, sep="", end="\n", flush=False):
        """Write a raw message (no timestamp or level prefix)."""
        message = sep.join(str(arg) for arg in args) + end
        self._write(message, flush)

    def info(self, *args, **kwargs):
        """Write an INFO-level timestamped message."""
        self._log_with_level("INFO", *args, **kwargs)

    def warning(self, *args, **kwargs):
        """Write a WARNING-level timestamped message."""
        self._log_with_level("WARNING", *args, **kwargs)

    def error(self, *args, **kwargs):
        """Write an ERROR-level timestamped message."""
        self._log_with_level("ERROR", *args, **kwargs)

    # ---------------- internal ---------------- #

    def _log_with_level(self, level, *args, sep=" ", end="\n", flush=False):
        """Build a ``[timestamp] [LEVEL] message`` line and send to ``_write``."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{ts}] [{level}] " + sep.join(str(a) for a in args) + end
        self._write(message, flush)

    def _write(self, message: str, flush: bool):
        """
        Thread-safe write to stdout and (if enabled) the log file.

        On file I/O errors, file logging is silently disabled after one
        warning to stderr.
        """
        with self._lock:
            # stdout
            sys.stdout.write(message)
            if flush:
                sys.stdout.flush()

            # file
            if not self._log_file or not self._file_enabled:
                return

            try:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(message)
                    if flush:
                        f.flush()
            except Exception as e:
                self._file_enabled = False
                if not self._warned:
                    self._warned = True
                    sys.stderr.write(
                        f"[Logger WARNING] Log file disabled due to error: {e}\n"
                    )


# -------- global singleton instance -------- #
log = _Logger()
