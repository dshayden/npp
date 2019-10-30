# import noisy libraries here without warnings
import logging
trimeshLog = logging.getLogger('trimesh')
trimeshLog.setLevel(logging.ERROR)

# Reduce extraneous tensorflow output
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# This is a private tensorflow API, don't use in general
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow

from . import SED
