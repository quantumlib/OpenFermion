# Install

Installing OpenFermion requires pip. Make sure that you are using an up-to-date
version of it. For information about getting started beyond what is provided
below please see our
[tutorial](https://github.com/quantumlib/OpenFermion/blob/master/examples/openfermion_tutorial.ipynb)
in the
[examples](https://github.com/quantumlib/OpenFermion/blob/master/examples)
folder as well as our detailed
[documentation](http://openfermion.readthedocs.io/en/latest/openfermion.html).

Currently, OpenFermion is only tested on Mac and Linux for the reason that both
electronic structure plugins are only compatible with Mac and Linux. However,
for those who would like to use Windows, or for anyone having other difficulties
with installing OpenFermion or its plugins, we have provided a Docker image and
usage instructions in the
[docker folder](https://github.com/quantumlib/OpenFermion/tree/master/docker).
The Docker image provides a virtual environment with OpenFermion and select
plugins pre-installed. The Docker installation should run on any operating
system.

You might also want to explore the alpha release of the
[OpenFermion Cloud Library](https://github.com/quantumlib/OpenFermion/tree/master/cloud_library)
where users can share and download precomputed molecular benchmark files.

Check out other [projects and papers](docs/projects.md) using OpenFermion for
inspiration, and let us know if you've been using OpenFermion!

## Developer install

To install the latest version of OpenFermion (in development mode):

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python -m pip install -e .
```

## Library install

To install the latest PyPI release as a library (in user mode):

```bash
python -m pip install --user openfermion
```
