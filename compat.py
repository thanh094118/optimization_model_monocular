import inspect
import sys
import numpy as np


def patch_numpy_and_inspect() -> None:
    """Compatibility patches for older libs under NumPy 2.x / Python 3.11+."""
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec

    for name in ["int", "float", "bool", "complex", "object", "str", "unicode"]:
        if not hasattr(np, name):
            setattr(np, name, str if name == "unicode" else eval(name))


def configure_stdout_encoding() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except OSError:
            pass
