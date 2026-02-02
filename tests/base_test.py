import unittest

from tests.util import Utils

class BaseTest(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        """Remove all __pycache__ folders after all tests in this class."""
        Utils.remove_all_pycache()
