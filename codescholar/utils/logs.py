import os
import sys

from loguru import logger

from codescholar.utils import paths

# remove any existing log handlers
logger.remove()

# Add to sys.stderr with our own configuration
logger.add(sys.stderr, level="INFO")

# Add to a file-log
logger.add(
    os.path.join(paths.get_logging_dir_path(), "codescholar_{time}.log"),
    retention=50,  # Max. 50 logs
    level="TRACE",
)
