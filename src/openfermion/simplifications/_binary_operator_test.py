import unittest
from _binary_operator import SymbolicBinary

class SymbolicBinaryTest(unittest.TestCase):

    def test_init_long_string(self):
        loc_op = SymbolicBinary('1 + w1 w2')
        self.assertEqual(loc_op.terms,[((1, '1'),), ((1, 'W'), (2, 'W'))])


    def test_init_tring(self):
        loc_op = SymbolicBinary('w1')
        self.assertEqual(loc_op.terms,[((1, 'W'),)])

    def test_init_list(self):
        loc_op = SymbolicBinary([((3, 'W'), (4, 'W'), (1, '1'))])
        self.assertEqual(loc_op.terms,[((3, 'W'), (4, 'W'))])

    def test_multiplication(self):
        loc_op1 = SymbolicBinary('1 + w1 w2')
        loc_op2 = SymbolicBinary([((3, 'W'), (4, 'W'), (1, '1'))])
        mult_op = loc_op1*loc_op2
        self.assertEqual(mult_op.terms,[((3, 'W'), (4, 'W')), ((1, 'W'), (2, 'W'), (3, 'W'), (4, 'W'))])

    def test_addition(self):
        loc_op1 = SymbolicBinary('w1 w2')
        loc_op2 = SymbolicBinary('1 + w1 w2')
        addn = loc_op1 + loc_op2
        self.assertEqual(addn.terms, [((1, '1'),)])

    def power_test(self):
        loc_op = SymbolicBinary('1 + w1 w2 + w3 w4')
        pow_loc = loc_op**2
        self.assertEqual(pow_loc.terms,[((1, '1'),), ((1, 'W'), (2, 'W')), ((3, 'W'), (4, 'W'))])

    def test_init_binary_rule(self):
        loc_op = SymbolicBinary('1 + w2 w2 + w2')
        self.assertEqual(loc_op.terms,[((1, '1'),)])