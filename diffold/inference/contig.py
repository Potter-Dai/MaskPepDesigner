"""Legacy import; implementation lives in MPDesigner.inference.contig."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("MPDesigner.inference.contig")
