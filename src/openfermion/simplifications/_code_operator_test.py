import unittest
from _code_operator import BinaryCode
from _binary_operator import SymbolicBinary

class BinaryCodeTest(unittest.TestCase):

    def test_basic_init(self):
        a = BinaryCode([[0, 1], [1, 0]], [SymbolicBinary(' w1 + w0 '), SymbolicBinary('w0 + 1')])
        self.assertEqual(a.dec[0].terms,[((1, 'W'),), ((0, 'W'),)])
        self.assertEqual(a.dec[1].terms,[((0, 'W'),), ((1, '1'),)])


