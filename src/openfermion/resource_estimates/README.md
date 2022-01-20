### Disclaimer: testing, dependencies, etc.

Code is system tested on Debian GNU/Linux with Python 3.8.5. All the code comes with tests (use `pytest`), but not unit tested with GitHub worflows.

Since the FT costing is closely tied to manipulation of molecular integrals (localization, active space selection, benchmarking against CCSD(T), ...) the code depends on [PySCF](https://pyscf.org/). Since we do not want to burden all OpenFermion users with this dependency, testing is disabled in the GitHub workflow. Moreover, the `resource_estimates` functionality requires the dependencies

```
pyscf
h5py~=3.3.0
jax
jaxlib
```

For THC factorization, it also requires [BTAS](https://github.com/ValeevGroup/BTAS) and the [PyBTAS](https://github.com/ncrubin/pybtas) wrapper, which require their own installation + depends.

### Overview

Module `openfermion.resource_estimates` to facilitate fault-tolerant (FT) resource estimates for chemical Hamiltonians.

The following factorizations are included:
* The [single](https://arxiv.org/abs/1902.02134) [factorization](https://arxiv.org/abs/1808.02625) (SF) method 
* The [double factorization](https://arxiv.org/pdf/2007.14460) (DF) method
* The [tensor hypercontraction](https://arxiv.org/abs/2011.03494) (THC) method

For the methods listed above, there are sub-routines which:
* factorize the two-electron integrals, `factorize()`
* compute the lambda values, `compute_lambda()`
* estimate the number of logical qubits and Toffoli gates required to simulate with this factorization, `compute_cost()`

There are also some costing routines for the [sparse factorization](https://arxiv.org/abs/1902.02134), but this is a work in progress.

### Details

The philosophy for this new module is to rely on PySCF to generate, store, and manipulate molecular information. The data (integrals, etc) is stored as a PySCF mean-field (`mf`) object. As an example, one could input an ionized water molecule like so:

```python
from pyscf import gto, scf

# input is just like any other PySCF script
mol = gto.M(
    atom = '''O    0.000000      -0.075791844    0.000000
              H    0.866811829    0.601435779    0.000000
              H   -0.866811829    0.601435779    0.000000
           ''',
    basis = 'augccpvtz',
    symmetry = False,
    charge = 1,
    spin = 1
)
mf = scf.ROHF(mol)
mf.verbose = 4
mf.kernel()  # run the SCF
```

Then, given the `mf` object, `resource_estimates.molecule` has routines to further manipulate the molecule, such as testing for stability (and reoptimizing), as well as localizing orbitals and performing automated active space selection with [AVAS](https://pubs.acs.org/doi/10.1021/acs.jctc.7b00128). Continuing our example:

```python
from openfermion.resource_estimates.molecule import stability, localize, avas_active_space

# make sure wave function is stable before we proceed
mf = stability(mf)

# localize before automatically selecting active space with AVAS
mf = localize(mf, loc_type='pm')  # default is loc_type ='pm' (Pipek-Mezey)

# you can use larger basis for `minao` to select non-valence...here select O 3s and 3p as well 
mol, mf = avas_active_space(mf, ao_list=['H 1s', 'O 2s', 'O 2p', 'O 3s', 'O 3p'], minao='ccpvtz') 
```

In each case, the input is the mean-field `mf` object, and the output is a modified `mf` object. The `mf` object is not updated in-place, so it is possible to create additional copies in memory.

At this point, we have a stable wave function, localized the orbitals, and selected an active space. At any point, the molecular Hamiltonian (e.g. active space) can be written out to HDF5 using `molecule.save_pyscf_to_casfile()`, or, if it exists, read in using `molecule.load_casfile_to_pyscf()`.

Once an active space is selected/generated, costing is relatively straightforward. There are helper functions for the SF, DF, and THC factorization schemes that will make a nice table given some parameters. For example:

```python
from openfermion.resource_estimates import sf

# make pretty SF costing table
sf.generate_costing_table(mf, name='water', rank_range=[20,25,30,35,40,45,50])
```
which outputs to a file called `single_factorization_water.txt`, and contains:

```
 Single low rank factorization data for 'water'.
    [*] using CAS((5a, 4b), 11o)
        [+]                      E(SCF):       -75.63393088
        [+] Active space CCSD(T) E(cor):        -0.08532629
        [+] Active space CCSD(T) E(tot):       -75.71925716
============================================================================================================
     L          ||ERI - SF||       lambda      CCSD(T) error (mEh)       logical qubits       Toffoli count    
------------------------------------------------------------------------------------------------------------
     20          1.7637e-01        212.7              -2.97                   298                4.3e+08       
     25          5.7546e-02        215.0               1.53                   298                4.7e+08       
     30          2.9622e-02        216.1               0.11                   298                5.1e+08       
     35          1.3728e-02        216.5              -0.07                   301                5.5e+08       
     40          2.1439e-03        216.7               0.00                   460                5.8e+08       
     45          2.8662e-04        216.8               0.00                   460                6.0e+08       
     50          1.1826e-04        216.8               0.00                   460                6.2e+08       
============================================================================================================
```

Note that the automated costing relies on error in CCSD(T)  - or CCSD, if desired - as the metric, so this may become a bottleneck for large active spaces.

The philosophy is that all costing methods are captured in the namespace related to the type of factorization (e.g., . So if one wanted to repeat the costing for DF or THC factorizations, one could 

```python
from openfermion.resource_estimates import df, thc

# make pretty DF costing table
df.generate_costing_table(mf, name='water', thresh_range=[1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5]) 

# make pretty THC costing table
# if you want to save each THC result to a file, you can set 'save_thc' to True
thc.generate_costing_table(mf, name='water', nthc_range=[20,25,30,35,40,45,50], save_thc=False) 
```

Which generate similar outputs, e.g. the above would generate tables in `double_factorization_water.txt` and `thc_factorization_water.txt`. 

More fine-grained control is given by subroutines that compute the factorization, the lambda values, and the cost estimates. For example, considering the double factorization, we could have

```python
factorized_eris, df_factors, _, _ = df.factorize(mf._eri, cutoff_threshhold)
df_lambda  = df.compute_lambda(mf, df_factors)
_, number_toffolis, num_logical_qubits = df.compute_cost(num_spin_orbitals, df_lambda, *args)
```
which, unlike the pretty tables above, require the user to handle and input several molecular quantities and intermediates, but at the gain of more functionality and control. Switching between factorization schemes is generally as easy as swapping out the namespace, for example to perform different factorizations on the ERIs, 

```python
sf.factorize()
df.factorize()
thc.factorize()
```

are all valid, as are the methods `compute_lambda()` and `compute_cost()` for the factorizations.


For THC factorization, it also requires [BTAS](https://github.com/ValeevGroup/BTAS) and the [PyBTAS](https://github.com/ncrubin/pybtas) wrapper, which require their own installation + depends.

Again, since we do not wish to burden all OpenFermion users with these dependencies, testing with GitHub workflows is disabled, but if you install the dependencies, running `pytest` should pass.
