import glob
import os
import unittest

from codescholar.utils import paths
from codescholar.utils.logs import logger


class LoggingTests(unittest.TestCase):
    def test_file_sink_1(self):
        log_dir_path = paths.get_logging_dir_path()

        msg = "Testing CodeScholar..."
        logger.info(msg)

        #  Check that logs exist
        self.assertTrue(os.path.exists(log_dir_path))

        def _read_path(_path: str):
            with open(_path, "r") as f:
                return f.read()

        self.assertTrue(any(msg in _read_path(path) for path in glob.glob(os.path.join(log_dir_path, "*.log"))))
