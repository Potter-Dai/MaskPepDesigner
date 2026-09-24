"""Legacy import; implementation lives in MPDesigner.config_v6."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("MPDesigner.config_v6")
