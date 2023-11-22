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
"""Dataclasses for resource estimation."""
from dataclasses import dataclass, asdict, field
import pandas as pd

from openfermion.resource_estimates.pbc.hamiltonian import HamiltonianProperties


@dataclass(frozen=True)
class ResourceEstimates:
    """Dataclass to hold return values from compute_cost functions.

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
        ham_props: list of lambda values for each cutoff.
        resources: list of resource estimates for each cutoff.
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
