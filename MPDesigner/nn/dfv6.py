"""Compatibility import for the pre-migration module name."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("MPDesigner.nn.mpdesigner")
