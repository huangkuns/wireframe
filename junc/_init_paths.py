import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = os.path.join(this_dir)
add_path(lib_path)
