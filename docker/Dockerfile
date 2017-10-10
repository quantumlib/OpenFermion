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

# Dockerfile for OpenFermion+ProjectQ (including plugin)

FROM ubuntu

USER root

RUN apt-get update

# Install python, pip
RUN yes | apt-get install wget
RUN cd /root; wget https://bootstrap.pypa.io/get-pip.py
## Change the following lines for different Python versions/distributions
RUN yes | apt-get install python
RUN python /root/get-pip.py

# Install projectq (login to Github may be requested)
RUN yes | apt-get install git
RUN cd /root; git clone https://github.com/ProjectQ-Framework/ProjectQ
RUN pip install -r /root/ProjectQ/requirements.txt
RUN yes | apt-get install gcc
RUN yes | apt-get install g++
RUN yes | apt-get install python-dev
RUN cd /root/ProjectQ/; python -m pip install -e .

# Install OpenFermion, OpenFermion-ProjectQ plugin
RUN pip install git+https://github.com/quantumlib/OpenFermion-ProjectQ.git@v0.1a1
RUN pip install git+https://github.com/ProjectQ-Framework/FermiLib-Plugin-Psi4.git@v0.1a1
RUN pip install git+https://github.com/ProjectQ-Framework/FermiLib-Plugin-PySCF.git@v0.1a0
