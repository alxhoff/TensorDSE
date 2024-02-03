"""
    Missing  Docstring: TODO
"""

import os

SPLITTER_DIR  = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR     = os.path.dirname(SPLITTER_DIR)
WORK_DIR      = os.path.dirname(UTILS_DIR)

RESOURCES_DIR = os.path.join(WORK_DIR, "resources")

MODELS_DIR    = os.path.join(SPLITTER_DIR, "models")
MAPPING_DIR   = os.path.join(SPLITTER_DIR, "mapping")

SOURCE_DIR    = os.path.join(MODELS_DIR, "source")
SUB_DIR       = os.path.join(MODELS_DIR, "sub")

LAYERS_DIR    = os.path.join(SUB_DIR, "tflite")
COMPILED_DIR  = os.path.join(SUB_DIR, "compiled")
