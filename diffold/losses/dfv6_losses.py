"""Legacy import; implementation lives in MPDesigner.losses.dfv6_losses."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("MPDesigner.losses.dfv6_losses")
