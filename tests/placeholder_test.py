import unittest

class PlaceHolderTest(unittest.TestCase):
    def placeholder_test(self):
        # some computation
        a, b, c = 1, 2, 3

        # assertion for test
        self.assertTrue(c == a+b)
