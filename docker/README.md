# Docker setup for OpenFermion and select plugins

This Docker image contains [OpenFermion](https://github.com/quantumlib/OpenFermion)
and its available plugins for
[Cirq](https://github.com/quantumlib/Cirq),
[Psi4](https://github.com/quantumlib/OpenFermion-Psi4), and
[PySCF](https://github.com/quantumlib/OpenFermion-PySCF).
Check out Docker's [website](https://www.docker.com/what-container) for a description of
what a container image is and why it can be so useful.
The Docker-based installation is extremely robust and runs on any operating
system, so it is an ideal solution for anyone having difficulty installing
OpenFermion (or any of its plugins) using the standard procedure.


## What's included?

- Git
- Python 3.12
- [Miniforge](https://github.com/conda-forge/miniforge)
- [OpenFermion](https://github.com/quantumlib/OpenFermion)
- [Cirq](https://github.com/quantumlib/Cirq)
- [Psi4](http://www.psicode.org)
- [PySCF](https://github.com/sunqm/pyscf)
- [OpenFermion-Cirq](https://github.com/quantumlib/OpenFermion-Cirq)
- [OpenFermion-Psi4](https://github.com/quantumlib/OpenFermion-Psi4)
- [OpenFermion-PySCF](https://github.com/quantumlib/OpenFermion-PySCF)


## Setting up Docker for the first time
The Dockerfile is based on the [Ubuntu image](https://hub.docker.com/_/ubuntu) (ver. 22.04).
Two Linux architectures are supported in the Dockerfile: x86_64 (amd64) and aarch64 (arm64).
Detecting your host's architecture is handled automatically through the following docker image building process, so you don't have to check it manually.
It creates a Python (ver. 3.12) virtual environment (named `fermion`) using Miniforge and installs all dependencies within it. Psi4 is installed with a [conda command](https://psicode.org/installs/v191/).

You can manually edit the Dockerfile if you need to set up a different development environment (e.g., changing the versions of Ubuntu, Python, Miniforge, Psi4, etc.).

### Building Docker Image
You first need to install [Docker](https://www.docker.com/).
Once Docker is setup, one can navigate to the folder containing the
Dockerfile for building the OpenFermion image (/docker/dockerfile) and run

```
docker build -t openfermion_docker .
```
where "openfermion_docker" is just an arbitrary name for our docker image.
Building the Dockerfile starts from a base image of Ubuntu and then installs
OpenFermion, its plugins, and the necessary applications needed for running these
programs. This is a fairly involved setup and will take some time
(perhaps up to thirty minutes, depending on the computer) and disk space (several gigabytes). 

Line 18 in the Dockerfile
```
COPY . /root/workspace
```
copy the files in the current local directory (where the Dockerfile is located) when the image is built.
If you don't want to copy the files, delete the line first and then build the image.

### Running the Container
Once the image has been built, run the image with
```
docker run -it --name openfermion_container -v $(pwd):/root/workspace openfermion_docker
```
where "openfermion_container" is an arbitrary choice for the name of our docker container. This command will mount your current local directory, where the Dockerfile is located, to `/root/workspace` inside the running container.
By default, the virtual environment `fermion` is automatically activated in the running container.

If you don't want to mount the current directory, run the following command instead:
```
docker run -it --name openfermion_container openfermion_docker
```

### Copy Local Files into the Container
Line 18 in the Dockerfile (`COPY . /root/workspace`) copies the current files only once when the image is built.
Local files in any directories can be copied using:
```
docker cp [path to file on disk] [container name]:[path in container]
```
An alternative way of loading files onto the Docker container is through
remote repos such as GitHub. Git is installed in the Docker image.
After `docker run`, one could run "git clone ..." etc to pull files
remotely into the Docker container.


## Running Jupyter notebook with Docker backend

To run Jupyter notebooks (such as our demos) in a browser with a Docker container
running as a backend, first check the ip address of the virtual machine by running

```
docker-machine ip default
```

where "default" can be replaced by the name of whichever virtual machine whose
ip address you want to check. Assuming the Docker image for OpenFermion is built
and called openfermion_docker, run the container with an additional -p flag:


```
docker run -it -p 8888:8888 openfermion_docker
```

Here the numbers 8888 simply specifies the port number through which the Docker
container communicates with the browser. If for some reason this port is not
available, any other number in 8000-9000 will do. When the terminal enters the Docker container,
run a Jupyter notebook with:

```
jupyter-notebook --allow-root --no-browser --port 8888 --ip=0.0.0.0
```

where 8888 is the port number used previously for setting up the container.
The message returned to the terminal should end with a statement that says
something like:
```
Copy/paste this URL into your browser when you connect for the first time,
to login with a token:
   http://0.0.0.0:8888/?token=8f70c035fb9b0dbbf160d996f7f341fecf94c9aedc7cfaf7
```

Note the token string 8f70c035fb9b0dbbf160d996f7f341fecf94c9aedc7cfaf7.
Open a browser and type in the address line

```
        [virtual machine ip]:8888
```

where [virtual machine ip] is extracted from `docker-machine ip` and 8888 is the port
number (or any other port number that one specifies previously). A webpage
asking for token string should appear. Use the token string obtained from before to
enter Jupyter notebook. If logged in successfully, you should be able to freely
navigate through the entire Docker image and launch any Jupyter notebook in the image.
