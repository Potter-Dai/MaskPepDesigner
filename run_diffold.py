"""Legacy entry point; use run_MPDesigner.py."""
import runpy
from pathlib import Path
from run_MPDesigner import *

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("run_MPDesigner.py")), run_name="__main__")
