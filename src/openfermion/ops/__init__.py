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
"""
This module should contain objects that describe the algebra of various
operators and representations which can be thought of as abstract instances that
serve as storage of the operators.  The abstract instance objects derive from
the PolynomialTensor.  Here we differentiate between generic storage objects and
particular instantiations.
"""
from .operators import (
    BinaryPolynomial,
    BinaryPolynomialError,
    BosonOperator,
    FermionOperator,
    IsingOperator,
    MajoranaOperator,
    QuadOperator,
    QubitOperator,
    SymbolicOperator,
    double_decoding,
    shift_decoder,
    BinaryCode,
    BinaryCodeError,
)

from .representations import (
    PolynomialTensor,
    PolynomialTensorError,
    general_basis_change,
    DiagonalCoulombHamiltonian,
    InteractionOperator,
    InteractionOperatorError,
    InteractionRDM,
    InteractionRDMError,
    QuadraticHamiltonian,
    QuadraticHamiltonianError,
    DOCIHamiltonian,
    antisymmetric_canonical_form,
)
