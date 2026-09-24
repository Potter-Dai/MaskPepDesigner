"""Legacy import; implementation lives in MPDesigner.data.dfv6_ops."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("MPDesigner.data.dfv6_ops")
