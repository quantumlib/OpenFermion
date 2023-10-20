# coverage:ignore
"""Test cases for util.py
"""
from openfermion.resource_estimates.utils import QR, QI, QR2, QI2, power_two


def test_QR():
    """Tests function QR which gives the minimum cost for a QROM over L values
    of size M.
    """
    # Tests checked against Mathematica noteboook `costingTHC.nb`
    # Arguments are otherwise random
    assert QR(12341234, 5670) == (6, 550042)
    assert QR(12201990, 520199) == (2, 4611095)


def test_QI():
    """Tests function QI which gives the minimum cost for inverse QROM over
    L values.
    """
    # Tests checked against Mathematica noteboook `costingTHC.nb`
    # Arguments are otherwise random
    assert QI(987654) == (10, 1989)
    assert QI(8052021) == (11, 5980)


def test_QR2():
    """Tests function QR2 which gives the minimum cost for a QROM with two
    registers.
    """
    # Tests checked against Mathematica noteboook `costingsf.nb`
    # Arguments are otherwise random
    assert QR2(12, 34, 81) == (2, 2, 345)
    assert QR2(712, 340111, 72345) == (4, 16, 8341481)


def test_QI2():
    """Tests function QI which gives the minimum cost for inverse QROM with
    two registers.
    """
    # Tests checked against Mathematica noteboook `costingsf.nb`
    # Arguments are otherwise random
    assert QI2(1234, 5678) == (32, 64, 5519)
    assert QI2(7120, 1340111) == (4, 32768, 204052)


def test_power_two():
    """Test for power_two(m) which returns power of 2 that is a factor of m"""
    try:
        power_two(-1234)
    except AssertionError:
        pass
    assert power_two(0) == 0
    assert power_two(2) == 1
    assert power_two(3) == 0
    assert power_two(104) == 3  # 2**3 * 13
    assert power_two(128) == 7  # 2**7
    assert power_two(393120) == 5  # 2**5 * 3**3 * 5 * 7 * 13
