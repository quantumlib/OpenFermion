# Resource Estimation for Periodic Systems 

Module `openfermion.resource_estimates.pbc` facilitates fault-tolerant (FT) resource estimates for second-quantized symmetry-adapted Hamiltonians with periodic boundary conditions (PBC).

The module provides symmetry adapted sparse, single, double and tensor hypercontraction representations of the Hamiltonians. 

For the methods listed above, there are sub-routines which:
* factorize the two-electron integrals if appropriate
* compute the associated lambda values, `compute_lambda()`
* estimate the number of logical qubits and Toffoli gates required to simulate with this factorization, `compute_cost()`

### Details

Given a pyscf scf calculation of a periodic system with k-points:

```python
from pyscf.pbc import gto, scf

cell = gto.Cell()
cell.atom = '''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-hf-rev'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 0
cell.build()

kmesh = [1, 1, 3]
kpts = cell.make_kpts(kmesh)
nkpts = len(kpts)
mf = scf.KRHF(cell, kpts).rs_density_fit()
mf.kernel()
```

then resource estimates for the SF, DF, and THC factorization schemes can be generated a range of cutoffs. For example:

```python
from openfermion.resource_estimates.pbc import sf

costs = sf.generate_costing_table(mf, name='carbon_diamond', naux_cutoffs=[20,25,30,35,40,45,50])
print(costs.to_string(index=False))
```
will generate a `pandas.DataFrame` of resource estimates for the single factorization Hamiltonian (`sf`) and MP2 correlation energies for the range of auxiliary dimension (`naux_cutoffs`).


Note that the automated costing computes the MP2 correlation energy error as a reference point for monitoring the convergence of the factorization with respect to sparsity or the size of auxiliary dimension. MP2 may be a poor model chemistry depending on the system, changing the option `energy_method = "CCSD"` will use CCSD instead, but this may become too expensive as the system size grows.

The philosophy is that all costing methods are captured in the namespace related to the type of factorization. So if one wanted to repeat the costing for DF or THC factorizations, one could do 

```python
from openfermion.resource_estimates.pbc import df, thc

# We need to specify eigenvalue threshold for second factorization.
df_cutoffs = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
df_table = df.generate_costing_table(
    mf,
    name='carbon-diamond',
    cutoffs=df_cutoffs)

# Specify THC rank parameter we wish to scan over.
# Note THC dimension M = thc_rank_param * N, N = number of spin orbitals in the
# unit cell. 
thc_rank_params = [2, 4, 6]
# if you want to save each THC result to a file, you can set 'save_thc' to True
thc_table = thc.generate_costing_table(mf, name='carbon-diamond', thc_rank_params=thc_rank_params) 
```

More fine-grained control is given by subroutines that compute the factorization, the lambda values, and the cost estimates.
Further details are provided in a [tutorial](./notebooks/resource_estimates.ipynb). Note the THC factorization is more involved and we refer the reader to [thc-tutorial](../../notebooks/isdf.ipynb) for further details.

Similar to the case of molecular resource estimation, we do not wish to burden all OpenFermion users with these dependencies, and testing with GitHub workflows is disabled. Currently we only check if pyscf is available. If it is then pytest will pick up the pbc module and run the tests. Note the tests can be quite slow due to the cost associated with building the integrals.

## Requirements
Requirements can be found in [resource_estimates.txt](../../../../dev_tools/requirements/deps/resource_estimates.txt)
```
pyscf
jax
jaxlib
ase
```