# OpenFermion

OpenFermion is an open source library for compiling and analyzing quantum
algorithms to simulate fermionic systems, including quantum chemistry. Among
other functionalities, this version features data structures and tools for
obtaining and manipulating representations of fermionic and qubit Hamiltonians.
For more information, see the
<a href="https://arxiv.org/abs/1710.07629" class="external">release paper</a>.

## Plugins

OpenFermion relies on modular plugin libraries for significant functionality.
Specifically, plugins are used to simulate and compile quantum circuits and to
perform classical electronic structure calculations. Follow the links below to
learn more!

### Circuit compilation and simulation plugins

* [Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion) to
  support integration with [Forest](https://www.rigetti.com/forest).
* [SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson) to support integration
  with [Strawberry Fields](https://github.com/XanaduAI/strawberryfields).

### Electronic structure package plugins

* [OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4) to support
  integration with [Psi4](http://psicode.org).
* [OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF) to support
  integration with [PySCF](https://github.com/sunqm/pyscf).
* [OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac) to support
  integration with [DIRAC](http://diracprogram.org/doku.php).
