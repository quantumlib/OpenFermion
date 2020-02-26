## Libraries, Projects and Papers using OpenFermion

In this document we assemble known libraries, projects and papers that use,
discuss, or contribute functionality to OpenFermion in order to both provide
example use cases and document the ways its helping to impact electronic
structure on quantum computers. If you'd like to include your own projects
or papers, please contact us or submit a pull request.


### Libraries

[OpenFermion-Cirq](https://github.com/quantumlib/OpenFermion-Cirq)

This plugin library integrates OpenFermion with Google's
[Cirq](https://github.com/quantumlib/Cirq) framework in order to compile
quantum simulation algorithms to NISQ circuits using Cirq.

[Forest-OpenFermion](https://github.com/rigetticomputing/forestopenfermion) 

This plugin library integrates OpenFermion with Rigetti's
[Forest](https://www.rigetti.com/forest) framework in order to compile
quantum simulation algorithms to NISQ circuits using Forest.

[PennyLane](http://github.com/XanaduAI/PennyLane)

The `pennylane.qchem` module integrates OpenFermion with Xanadu's
[PennyLane](https://pennylane.ai) framework, allowing optimization
of quantum simulation algorithms using TensorFlow and PyTorch
on quantum hardware.

[SFOpenBoson](https://github.com/XanaduAI/SFOpenBoson)

This plugin library integrates OpenFermion with Xanadu's
[Strawberry Fields](https://github.com/XanaduAI/strawberryfields)
framework in order to compile quantum simulation algorithms that
pertain to the simulation of bosons.

[OpenFermion-Psi4](http://github.com/quantumlib/OpenFermion-Psi4)

This plugin library integrates OpenFermion with the electronic structure
package [Psi4](http://psicode.org).

[OpenFermion-PySCF](http://github.com/quantumlib/OpenFermion-PySCF)

This plugin library integrates OpenFermion with the electronic structure package
[PySCF](https://github.com/sunqm/pyscf).

[OpenFermion-Dirac](https://github.com/bsenjean/Openfermion-Dirac)

This plugin library integrates OpenFermion with the electronic structure package
[DIRAC](http://diracprogram.org/doku.php)

[OpenFermion-ProjectQ](https://github.com/quantumlib/OpenFermion-ProjectQ) - Deprecated

While no longer actively developed, this plugin library integrates OpenFermion
with the [ProjectQ](https://projectq.ch) framework for simulating quantum
circuits.


### Projects

[Hackathon Quantum Autoencoder](https://github.com/hsim13372/QCompress)

The winning project of the Rigetti quantum computing hackathon that combined
OpenFermion with Rigetti's framework, compressing molecular representations
with an autoencoder.

[CUSP Implementation](https://github.com/zapatacomputing/cusp_cirq_demo)

Implementation of Compressed Unsupervised State Preparation (CUSP) protocol
using OpenFermion and Cirq. CUSP uses the quantum autoencoder to synthesize
more compact circuits to use for algorithms such as VQE.


### Papers (we refer to only the arXiv versions) - last updated 9/3/18

***Calculating energy derivatives for quantum chemistry on a quantum computer***.
Thomas O'Brien, Bruno Senjean, Ramiro Sagastizabal, Xavier Bonet-Monroig,
Alicja Dutkiewicz, Francesco Buda, Leonardo DiCarlo and Lucas Visscher.
[arXiv:1905.03742](https://arxiv.org/abs/1905.03742). 2019.

***Strategies for Quantum Computing Molecular Energies using the Unitary Coupled
Cluster Ansatz***. Jonathan Romero, Ryan Babbush, Jarrod McClean, Cornelius
Hempel, Peter Love and Alán Aspuru-Guzik.
[arXiv:1701.02691](https://arxiv.org/abs/1701.02691). 2017.

***Low Depth Quantum Simulation of Electronic Structure***. Ryan Babbush, Nathan Wiebe,
Jarrod McClean, James McClain, Hartmut Neven and Garnet Chan.
[arXiv:1706.00023](https://arxiv.org/abs/1706.00023). 2017.

***A Language and Hardware Independent Approach to Quantum-Classical
Computing***. Alexander McCaskey, Eugene Dumitrescu, Dmitry Liakh, Mengsu Chen,
Wu-Chen Feng and Travis Humble.
[arXiv:1710.01794](https://arxiv.org/abs/1710.01794). 2017.

***OpenFermion: The Electronic Structure Package for Quantum Computers***.
Jarrod McClean, Ian Kivlichan, Kevin Sung, Damian Steiger, Yudong Cao, Chengyu Dai,
E. Schuyler Fried, Craig Gidney, Brendan Gimby, Pranav Gokhale, Thomas Häner,
Tarini Hardikar, Vojtĕch Havlíček, Cupjin Huang, Josh Izaac, Zhang Jiang, Xinle Liu,
Matthew Neeley, Thomas O'Brien, Isil Ozfidan, Maxwell Radin, Jhonathan Romero,
Nicholas Rubin, Nicolas Sawaya, Kanav Setia, Sukin Sim, Mark Steudtner,
Qiming Sun, Wei Sun, Fang Zhang and Ryan Babbush.
[arXiv:1710.07629](https://arxiv.org/abs/1710.07629). 2017.

***Quantum Simulation of Electronic Structure with Linear Depth and Connectivity***.
Ian Kivlichan, Jarrod McClean, Nathan Wiebe, Craig Gidney,
Alán Aspuru-Guzik, Garnet Chan and Ryan Babbush.
[arXiv:1711.04789](https://arxiv.org/abs/1711.04789). 2017.

***Quantum Algorithms to Simulate Many-Body Physics of Correlated Fermions***.
Zhang Jiang, Kevin Sung, Kostyantyn Kechedzhi, Vadim Smelyanskiy and Sergio Boixo.
[arXiv:1711.05395](https://arxiv.org/abs/1711.05395). 2017.

***Improved Techniques for Preparing Eigenstates of Fermionic Hamiltonians***.
Dominic Berry, Mária Kieferová, Artur Scherer, Yuval Sanders,
Guang Hao Low, Nathan Wiebe, Craig Gidney and Ryan Babbush.
[arXiv:1711.10460](https://arxiv.org/abs/1711.10460). 2017.

***Lowering Qubit Requirements for Quantum Simulation of Fermionic Systems***.
Mark Steudtner and Stephanie Wehner.
[arXiv:1712.07067](https://arxiv.org/abs/1712.07067). 2017.

***Bravyi-Kitaev Superfast Simulation of Fermions on a Quantum Computer***.
Kanav Setia and James Whitfield.
[arXiv:1712.00446](https://arxiv.org/abs/1712.00446). 2017.

***Application of Fermionic Marginal Constraints to Hybrid Quantum
Algorithms***. Nicholas Rubin, Ryan Babbush and Jarrod McClean.
[arXiv:1801.03524](https://arxiv.org/abs/1801.03524). 2018.

***Cloud Quantum Computing of an Atomic Nucleus***.
E. Dumitrescu, A. McCaskey, G. Hagen, G. Jansen, T. Morris,
T. Papenbrock, R. Pooser, D. Dean and P. Lougovski.
[arXiv:1801.03897](https://arxiv.org/abs/1801.03897). 2018.

***Quantum Chemistry Calculations on a Trapped-Ion Quantum Simulator***.
Cornelius Hempel, Christine Maier, Jonathan Romero, Jarrod McClean,
Thomas Monz, Heng Shen, Petar Jurcevic, Ben Lanyon, Peter Love, Ryan Babbush,
Alán Aspuru-Guzik, Rainer Blatt and Christian Roos.
[arXiv:1803.10238](https://arxiv.org/abs/1803.10238). 2018.

***Strawberry Fields: A Software Platform for Photonic Quantum Computing***.
Nathan Killoran, Josh Izaac, Nicolás Quesada, Ville Bergholm, Matthew Amy and
Christian Weedbrook.
[arXiv:1804.03159](https://arxiv.org/abs/1804.03159). 2018.

***Accounting for Errors in Quantum Algorithms via Individual Error
Reduction***. Matthew Otten and Stephen Gray.
[arXiv:1804.06969](https://arxiv.org/abs/1804.06969). 2018.

***Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity***.
Ryan Babbush, Craig Gidney, Dominic Berry, Nathan Wiebe, Jarrod McClean,
Alexandru Paler, Austin Fowler and Hartmut Neven.
[arXiv:1805.03662](https://arxiv.org/abs/1805.03662). 2018.

***A Universal Quantum Computing Virtual Machine***.
Qian-Tan Hong, Zi-Yong Ge, Wen Wang, Hai-Feng Lang, Zheng-An Wang, Yi Peng,
Jin-Jun Chen, Li-Hang Ren, Yu Zeng, Liang-Zhu Mu and Heng Fan.
[arXiv:1806.06511](https://arxiv.org/abs/1806.06511). 2018.

***Overview and Comparison of Gate Level Quantum Software Platforms***.
Ryan LaRose. [arXiv:1807.02500](https://arxiv.org/abs/1807.02500). 2018.

***Low Rank Representations for Quantum Simulation of Electronic Structure***.
Mario Motta, Erika Ye, Jarrod McClean, Zhendong Li, Austin Minnich,
Ryan Babbush and Garnet Chan.
[arXiv:1808.02625](https://arxiv.org/abs/1808.02625). 2018.

***Quantum Computational Chemistry***.
Sam McArdle, Suguru Endo, Alán Aspuru-Guzik, Simon Benjamin and Xiao Yuan.
[arXiv:1808.10402](https://arxiv.org/abs/1808.10402). 2018.

***Quantum Phase Estimation for Noisy, Smalle-Scale Experiment***.
Thomas O'Brien, Brian Tarasinski and Barbara Terhal.
[arXiv:1809.09697](https://arxiv.org/abs/1809.09697). 2018.
