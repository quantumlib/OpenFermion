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

# Dockerfile for OpenFermion, Cirq, and select plugins.

FROM ubuntu

USER root

RUN apt-get update

# Install utilities
RUN apt-get install -y bzip2
RUN apt-get install -y cmake
RUN apt-get install -y git
RUN apt-get install -y wget
RUN apt-get install -y libblas-dev
RUN apt-get install -y liblapack-dev

# Install Python 3
RUN apt-get install -y python3

# Install pip.
RUN apt-get install -y python3-pip

# Install Psi4.
RUN cd /root; wget http://vergil.chemistry.gatech.edu/psicode-download/psi4conda-1.2.1-py36-Linux-x86_64.sh
RUN echo '/root/psi4conda' | bash /root/psi4conda-1.2.1-py36-Linux-x86_64.sh
RUN rm /root/psi4conda-1.2.1-py36-Linux-x86_64.sh
RUN export PATH=/root/psi4conda/bin:$PATH

# Install PySCF.
RUN cd /root; git clone https://github.com/sunqm/pyscf
RUN cd /root/pyscf/pyscf/lib; mkdir build; cd build; cmake ..; make

# Install OpenFermion, Cirq, and plugins.
RUN pip3 install openfermion
RUN pip3 install cirq
RUN pip3 install openfermioncirq
RUN pip3 install openfermionpsi4
RUN pip3 install openfermionpyscf

# Update paths
RUN export PATH=/root/psi4conda/bin:$PATH
RUN export PYTHONPATH=/root/pyscf:$PYTHONPATH

# Make python point to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

ENTRYPOINT bash
