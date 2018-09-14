#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Tests for Hubbard model module."""

from openfermion.hamiltonians import bose_hubbard, fermi_hubbard


def test_fermi_hubbard_2x2_spinless():
    hubbard_model = fermi_hubbard(
            2, 2, 1.0, 4.0,
            chemical_potential=0.5,
            spinless=True)
    assert str(hubbard_model).strip() == """
-0.5 [0^ 0] +
4.0 [0^ 0 1^ 1] +
4.0 [0^ 0 2^ 2] +
-1.0 [0^ 1] +
-1.0 [0^ 2] +
-1.0 [1^ 0] +
-0.5 [1^ 1] +
4.0 [1^ 1 3^ 3] +
-1.0 [1^ 3] +
-1.0 [2^ 0] +
-0.5 [2^ 2] +
4.0 [2^ 2 3^ 3] +
-1.0 [2^ 3] +
-1.0 [3^ 1] +
-1.0 [3^ 2] +
-0.5 [3^ 3]
""".strip()

def test_fermi_hubbard_2x3_spinless():
    hubbard_model = fermi_hubbard(
            2, 3, 1.0, 4.0,
            chemical_potential=0.5,
            spinless=True)
    assert str(hubbard_model).strip() == """
-0.5 [0^ 0] +
4.0 [0^ 0 1^ 1] +
4.0 [0^ 0 2^ 2] +
-1.0 [0^ 1] +
-1.0 [0^ 2] +
-1.0 [0^ 4] +
-1.0 [1^ 0] +
-0.5 [1^ 1] +
4.0 [1^ 1 3^ 3] +
-1.0 [1^ 3] +
-1.0 [1^ 5] +
-1.0 [2^ 0] +
-0.5 [2^ 2] +
4.0 [2^ 2 3^ 3] +
4.0 [2^ 2 4^ 4] +
-1.0 [2^ 3] +
-1.0 [2^ 4] +
-1.0 [3^ 1] +
-1.0 [3^ 2] +
-0.5 [3^ 3] +
4.0 [3^ 3 5^ 5] +
-1.0 [3^ 5] +
-1.0 [4^ 0] +
-1.0 [4^ 2] +
-0.5 [4^ 4] +
4.0 [4^ 4 0^ 0] +
4.0 [4^ 4 5^ 5] +
-1.0 [4^ 5] +
-1.0 [5^ 1] +
-1.0 [5^ 3] +
-1.0 [5^ 4] +
-0.5 [5^ 5] +
4.0 [5^ 5 1^ 1]
""".strip()

def test_fermi_hubbard_3x2_spinless():
    hubbard_model = fermi_hubbard(
            3, 2, 1.0, 4.0,
            chemical_potential=0.5,
            spinless=True)
    assert str(hubbard_model).strip() == """
-0.5 [0^ 0] +
4.0 [0^ 0 1^ 1] +
4.0 [0^ 0 3^ 3] +
-1.0 [0^ 1] +
-1.0 [0^ 2] +
-1.0 [0^ 3] +
-1.0 [1^ 0] +
-0.5 [1^ 1] +
4.0 [1^ 1 2^ 2] +
4.0 [1^ 1 4^ 4] +
-1.0 [1^ 2] +
-1.0 [1^ 4] +
-1.0 [2^ 0] +
-1.0 [2^ 1] +
-0.5 [2^ 2] +
4.0 [2^ 2 0^ 0] +
4.0 [2^ 2 5^ 5] +
-1.0 [2^ 5] +
-1.0 [3^ 0] +
-0.5 [3^ 3] +
4.0 [3^ 3 4^ 4] +
-1.0 [3^ 4] +
-1.0 [3^ 5] +
-1.0 [4^ 1] +
-1.0 [4^ 3] +
-0.5 [4^ 4] +
4.0 [4^ 4 5^ 5] +
-1.0 [4^ 5] +
-1.0 [5^ 2] +
-1.0 [5^ 3] +
-1.0 [5^ 4] +
-0.5 [5^ 5] +
4.0 [5^ 5 3^ 3]
""".strip()

def test_fermi_hubbard_2x2_spinful():
    hubbard_model = fermi_hubbard(
            2, 2, 1.0, 4.0,
            chemical_potential=0.5,
            magnetic_field=0.3,
            spinless=False)
    assert str(hubbard_model).strip() == """
-0.8 [0^ 0] +
4.0 [0^ 0 1^ 1] +
-1.0 [0^ 2] +
-1.0 [0^ 4] +
-0.2 [1^ 1] +
-1.0 [1^ 3] +
-1.0 [1^ 5] +
-1.0 [2^ 0] +
-0.8 [2^ 2] +
4.0 [2^ 2 3^ 3] +
-1.0 [2^ 6] +
-1.0 [3^ 1] +
-0.2 [3^ 3] +
-1.0 [3^ 7] +
-1.0 [4^ 0] +
-0.8 [4^ 4] +
4.0 [4^ 4 5^ 5] +
-1.0 [4^ 6] +
-1.0 [5^ 1] +
-0.2 [5^ 5] +
-1.0 [5^ 7] +
-1.0 [6^ 2] +
-1.0 [6^ 4] +
-0.8 [6^ 6] +
4.0 [6^ 6 7^ 7] +
-1.0 [7^ 3] +
-1.0 [7^ 5] +
-0.2 [7^ 7]
""".strip()

def test_fermi_hubbard_2x2_spinful_phs():
    hubbard_model = fermi_hubbard(
            2, 2, 1.0, 4.0,
            chemical_potential=0.5,
            magnetic_field=0.3,
            spinless=False,
            particle_hole_symmetry=True)
    assert str(hubbard_model).strip() == """
4.0 [] +
-2.8 [0^ 0] +
4.0 [0^ 0 1^ 1] +
-1.0 [0^ 2] +
-1.0 [0^ 4] +
-2.2 [1^ 1] +
-1.0 [1^ 3] +
-1.0 [1^ 5] +
-1.0 [2^ 0] +
-2.8 [2^ 2] +
4.0 [2^ 2 3^ 3] +
-1.0 [2^ 6] +
-1.0 [3^ 1] +
-2.2 [3^ 3] +
-1.0 [3^ 7] +
-1.0 [4^ 0] +
-2.8 [4^ 4] +
4.0 [4^ 4 5^ 5] +
-1.0 [4^ 6] +
-1.0 [5^ 1] +
-2.2 [5^ 5] +
-1.0 [5^ 7] +
-1.0 [6^ 2] +
-1.0 [6^ 4] +
-2.8 [6^ 6] +
4.0 [6^ 6 7^ 7] +
-1.0 [7^ 3] +
-1.0 [7^ 5] +
-2.2 [7^ 7]
""".strip()

def test_fermi_hubbard_2x2_spinful_aperiodic():
    hubbard_model = fermi_hubbard(
            2, 2, 1.0, 4.0,
            chemical_potential=0.5,
            magnetic_field=0.3,
            spinless=False,
            periodic=False)
    assert str(hubbard_model).strip() == """
-0.8 [0^ 0] +
4.0 [0^ 0 1^ 1] +
-1.0 [0^ 2] +
-1.0 [0^ 4] +
-0.2 [1^ 1] +
-1.0 [1^ 3] +
-1.0 [1^ 5] +
-1.0 [2^ 0] +
-0.8 [2^ 2] +
4.0 [2^ 2 3^ 3] +
-1.0 [2^ 6] +
-1.0 [3^ 1] +
-0.2 [3^ 3] +
-1.0 [3^ 7] +
-1.0 [4^ 0] +
-0.8 [4^ 4] +
4.0 [4^ 4 5^ 5] +
-1.0 [4^ 6] +
-1.0 [5^ 1] +
-0.2 [5^ 5] +
-1.0 [5^ 7] +
-1.0 [6^ 2] +
-1.0 [6^ 4] +
-0.8 [6^ 6] +
4.0 [6^ 6 7^ 7] +
-1.0 [7^ 3] +
-1.0 [7^ 5] +
-0.2 [7^ 7]
""".strip()


def test_bose_hubbard_2x2():
    hubbard_model = bose_hubbard(
        2, 2, 1.0, 4.0,
        chemical_potential=0.5,
        dipole=0.3)
    assert str(hubbard_model).strip() == """
-1.0 [0 1^] +
-1.0 [0 2^] +
-2.5 [0^ 0] +
2.0 [0^ 0 0^ 0] +
0.3 [0^ 0 1^ 1] +
0.3 [0^ 0 2^ 2] +
-1.0 [0^ 1] +
-1.0 [0^ 2] +
-1.0 [1 3^] +
-2.5 [1^ 1] +
2.0 [1^ 1 1^ 1] +
0.3 [1^ 1 3^ 3] +
-1.0 [1^ 3] +
-1.0 [2 3^] +
-2.5 [2^ 2] +
2.0 [2^ 2 2^ 2] +
0.3 [2^ 2 3^ 3] +
-1.0 [2^ 3] +
-2.5 [3^ 3] +
2.0 [3^ 3 3^ 3]
""".strip()

def test_bose_hubbard_2x3():
    hubbard_model = bose_hubbard(
        2, 3, 1.0, 4.0,
        chemical_potential=0.5,
        dipole=0.3)
    assert str(hubbard_model).strip() == """
-1.0 [0 1^] +
-1.0 [0 2^] +
-1.0 [0 4^] +
-2.5 [0^ 0] +
2.0 [0^ 0 0^ 0] +
0.3 [0^ 0 1^ 1] +
0.3 [0^ 0 2^ 2] +
-1.0 [0^ 1] +
-1.0 [0^ 2] +
-1.0 [0^ 4] +
-1.0 [1 3^] +
-1.0 [1 5^] +
-2.5 [1^ 1] +
2.0 [1^ 1 1^ 1] +
0.3 [1^ 1 3^ 3] +
-1.0 [1^ 3] +
-1.0 [1^ 5] +
-1.0 [2 3^] +
-1.0 [2 4^] +
-2.5 [2^ 2] +
2.0 [2^ 2 2^ 2] +
0.3 [2^ 2 3^ 3] +
0.3 [2^ 2 4^ 4] +
-1.0 [2^ 3] +
-1.0 [2^ 4] +
-1.0 [3 5^] +
-2.5 [3^ 3] +
2.0 [3^ 3 3^ 3] +
0.3 [3^ 3 5^ 5] +
-1.0 [3^ 5] +
-1.0 [4 5^] +
-2.5 [4^ 4] +
0.3 [4^ 4 0^ 0] +
2.0 [4^ 4 4^ 4] +
0.3 [4^ 4 5^ 5] +
-1.0 [4^ 5] +
-2.5 [5^ 5] +
0.3 [5^ 5 1^ 1] +
2.0 [5^ 5 5^ 5]
""".strip()

def test_bose_hubbard_3x2():
    hubbard_model = bose_hubbard(
        3, 2, 1.0, 4.0,
        chemical_potential=0.5,
        dipole=0.3)
    assert str(hubbard_model).strip() == """
-1.0 [0 1^] +
-1.0 [0 2^] +
-1.0 [0 3^] +
-2.5 [0^ 0] +
2.0 [0^ 0 0^ 0] +
0.3 [0^ 0 1^ 1] +
0.3 [0^ 0 3^ 3] +
-1.0 [0^ 1] +
-1.0 [0^ 2] +
-1.0 [0^ 3] +
-1.0 [1 2^] +
-1.0 [1 4^] +
-2.5 [1^ 1] +
2.0 [1^ 1 1^ 1] +
0.3 [1^ 1 2^ 2] +
0.3 [1^ 1 4^ 4] +
-1.0 [1^ 2] +
-1.0 [1^ 4] +
-1.0 [2 5^] +
-2.5 [2^ 2] +
0.3 [2^ 2 0^ 0] +
2.0 [2^ 2 2^ 2] +
0.3 [2^ 2 5^ 5] +
-1.0 [2^ 5] +
-1.0 [3 4^] +
-1.0 [3 5^] +
-2.5 [3^ 3] +
2.0 [3^ 3 3^ 3] +
0.3 [3^ 3 4^ 4] +
-1.0 [3^ 4] +
-1.0 [3^ 5] +
-1.0 [4 5^] +
-2.5 [4^ 4] +
2.0 [4^ 4 4^ 4] +
0.3 [4^ 4 5^ 5] +
-1.0 [4^ 5] +
-2.5 [5^ 5] +
0.3 [5^ 5 3^ 3] +
2.0 [5^ 5 5^ 5]
""".strip()

def test_bose_hubbard_2x2_aperiodic():
    hubbard_model = bose_hubbard(
        2, 2, 1.0, 4.0,
        chemical_potential=0.5,
        dipole=0.3,
        periodic=False)
    assert str(hubbard_model).strip() == """
-1.0 [0 1^] +
-1.0 [0 2^] +
-2.5 [0^ 0] +
2.0 [0^ 0 0^ 0] +
0.3 [0^ 0 1^ 1] +
0.3 [0^ 0 2^ 2] +
-1.0 [0^ 1] +
-1.0 [0^ 2] +
-1.0 [1 3^] +
-2.5 [1^ 1] +
2.0 [1^ 1 1^ 1] +
0.3 [1^ 1 3^ 3] +
-1.0 [1^ 3] +
-1.0 [2 3^] +
-2.5 [2^ 2] +
2.0 [2^ 2 2^ 2] +
0.3 [2^ 2 3^ 3] +
-1.0 [2^ 3] +
-2.5 [3^ 3] +
2.0 [3^ 3 3^ 3]
""".strip()
