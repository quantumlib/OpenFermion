# coverage: ignore
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
import os

from pyscf.pbc import gto, scf
from pyscf.lib import chkfile


def make_diamond_113_szv() -> scf.KRHF:
    """Build diamond in gth-szv basis with 113 k-point mesh from checkpoint
    file.
    """
    TEST_CHK = os.path.join(os.path.dirname(__file__), "../test_data/scf.chk")
    cell = gto.Cell()
    cell.atom = """
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    """
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf-rev"
    cell.a = """
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000"""
    cell.unit = "B"
    cell.verbose = 0
    cell.build(parse_arg=False)

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    scf_dict = chkfile.load(TEST_CHK, "scf")
    mf.__dict__.update(scf_dict)
    mf.with_df._cderi = TEST_CHK
    mf.with_df.mesh = cell.mesh
    return mf
