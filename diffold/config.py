"""Legacy import; implementation lives in MPDesigner.config."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("MPDesigner.config")
