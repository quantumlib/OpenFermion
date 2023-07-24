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
"""Function defining beta parameter for controlled rotations in DF and THC"""
import numpy as np


def compute_beta_for_resources(num_spin_orbs: int, num_kpts: int,
                               de_for_qpe: float):
    """Compute beta (number of bits for controlled rotations).
    
    Uses expression from https://arxiv.org/pdf/2007.14460.pdf.

    Args:
        num_spin_orbs: Number of spin orbitals (per k-point)
        num_kpts: Number of k-points.
        de_for_qpe: epsilon for phase estimation.
    """
    return np.ceil(5.652 + np.log2(num_spin_orbs * num_kpts / de_for_qpe))
