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
from dataclasses import dataclass, asdict, field
import numpy as np
import pandas as pd

from openfermion.resource_estimates.pbc.utils.hamiltonian_utils import (
    HamiltonianProperties,)


@dataclass(frozen=True)
class ResourceEstimates:
    """Lighweight descriptive data class to hold return values from compute_cost
    functions.

    Attributes:
        toffolis_per_step: Toffolis per step
        total_toffolis: Total number of Toffolis
        logical_qubits: Total ancilla cost
    """

    toffolis_per_step: int
    total_toffolis: int
    logical_qubits: int

    dict = asdict


@dataclass
class PBCResources:
    """Helper class to hold resource estimates given a range of cutoffs.

    Attributes:
        system_name: Descriptive name for calculation.
        num_spin_orbitals: Number of spin orbitals.
        num_kpts: Number of k-points.
        dE: Epsilon for phase estimation.
        chi: A number of bits controlling the precision of the factorization.
            What this means is dependent on the factorization.
        exact_energy: Exact energy (no cutoff).
        energy_method: Which method to use to copute energy (MP2 or CCSD).
        cutoff: list of cuttofs.
        ham_properties: list of lambda values for each cutoff.
        resource_estimates: list of resource estimates for each cutoff.
    """

    system_name: str
    num_spin_orbitals: int
    num_kpts: int
    dE: float
    chi: int
    exact_energy: float
    energy_method: str = "MP2"
    cutoff: list = field(default_factory=list)
    approx_energy: list = field(default_factory=list)
    ham_props: list = field(default_factory=list)
    resources: list = field(default_factory=list)

    dict = asdict

    def to_dataframe(self) -> pd.DataFrame:
        """Convert PBCResources instance to pandas DataFrame."""
        df = pd.DataFrame(self.dict())
        lambdas = pd.json_normalize(df.pop("ham_props"))
        resources = pd.json_normalize(df.pop("resources"))
        df = df.join(pd.DataFrame(lambdas))
        df = df.join(pd.DataFrame(resources))
        return df

    def add_resources(
            self,
            ham_properties: HamiltonianProperties,
            resource_estimates: ResourceEstimates,
            cutoff: float,
            approx_energy: float,
    ) -> None:
        """Add resource estimates to container for given cutoff value.

        Args:
            ham_properties: lambda values
            resource_estimates: resource estimates for this value of cutoff
            cutoff: Current factorization/representation cutoff value.
            approx_energy: Current approximate energy estimate given cutoff.
        """
        self.ham_props.append(ham_properties)
        self.resources.append(resource_estimates)
        self.cutoff.append(cutoff)
        self.approx_energy.append(approx_energy)


def compute_beta_for_resources(num_spin_orbs, num_kpts, dE_for_qpe):
    """Compute beta (number of bits for rotations) using expression from
    https://arxiv.org/pdf/2007.14460.pdf.

    Args:
        num_spin_orbs: Number of spin orbitals
        num_kpts: Number of k-points.
        dE_for_qpe: epsilon for phase estimation.
    """
    return np.ceil(5.652 + np.log2(num_spin_orbs * num_kpts / dE_for_qpe))


def QR3(L, M1):
    r"""
    QR[Ll_, m_] := Ceiling[MinValue[{Ll/2^k + m*(2^k - 1), k >= 0}, k
    \[Element] Integers]];
    """
    k = 0.5 * np.log2(L / M1)
    value = lambda k: L / np.power(2, k) + M1 * (np.power(2, k) - 1)
    try:
        assert k >= 0
    except AssertionError:
        k_opt = 0
        val_opt = np.ceil(value(k_opt))
        assert val_opt.is_integer()
        return int(k_opt), int(val_opt)
    k_int = [np.floor(k), np.ceil(k)]  # restrict optimal k to integers
    k_opt = k_int[np.argmin(value(k_int))]  # obtain optimal k
    val_opt = np.ceil(value(k_opt))  # obtain ceiling of optimal value given k
    assert k_opt.is_integer()
    assert val_opt.is_integer()
    return int(k_opt), int(val_opt)


def QR2(L1, L2, M):
    """
     Table[Ceiling[L1/2^k1]*Ceiling[L2/2^k2] + M*(2^(k1 + k2) - 1), {k1, 1,
    10}, {k2, 1, 10}]
    """
    min_val = np.inf
    for k1 in range(1, 11):
        for k2 in range(1, 11):
            test_val = np.ceil(L1 /
                               (2**k1)) * np.ceil(L2 /
                                                  (2**k2)) + M * (2**
                                                                  (k1 + k2) - 1)
            if test_val < min_val:
                min_val = test_val
    return int(min_val)


def QI2(L1, Lv2):
    """
    QI2[L1_, L2_] :=
    Min[Table[
    Ceiling[L1/2^k1]*Ceiling[L2/2^k2] + 2^(k1 + k2), {k1, 1, 10}, {k2,
      1, 10}]];
    """
    min_val = np.inf
    for k1 in range(1, 11):
        for k2 in range(1, 11):
            test_val = np.ceil(L1 / (2**k1)) * np.ceil(Lv2 /
                                                       (2**k2)) + 2**(k1 + k2)
            if test_val < min_val:
                min_val = test_val
    return int(min_val)
