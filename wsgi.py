"""
Gunicorn entrypoint. main.py lives in app/ and uses relative imports
("import model_detect"), so app/ is added to sys.path before importing it,
rather than relying on gunicorn's --chdir path behavior.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))

from main import app  # noqa: E402
