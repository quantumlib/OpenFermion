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
- Python 3
- [OpenFermion](https://github.com/quantumlib/OpenFermion)
- [Cirq](https://github.com/quantumlib/Cirq)
- [Psi4](http://www.psicode.org)
- [PySCF](https://github.com/sunqm/pyscf)
- [OpenFermion-Cirq](https://github.com/quantumlib/OpenFermion-Cirq)
- [OpenFermion-Psi4](https://github.com/quantumlib/OpenFermion-Psi4)
- [OpenFermion-PySCF](https://github.com/quantumlib/OpenFermion-PySCF)


## Setting up Docker for the first time

You first need to install [Docker](https://www.docker.com/).
Once Docker is setup, one can navigate to the folder containing the
Dockerfile for building the OpenFermion image (docker/dockerfile) and run

```
docker build -t openfermion_docker .
```

where "openfermion_docker" is just an arbitrary name for our docker image.
Building the Dockerfile starts from a base image of Ubuntu and then installs
OpenFermion, its plugins, and the necessary applications needed for running these
programs. This is a fairly involved setup and will take some time
(perhaps up to thiry minutes depending on the computer). Once installation has
completed, run the image with

```
docker run -it openfermion_docker
```

With this command the terminal enters a new environment which emulates Ubuntu with
OpenFermion and accessories installed. To transfer files from somewhere on the disk to the Docker
container, first run `docker ps` in a seperate terminal from the one running
Docker. This returns a list of running containers, e.g.:

```
+CONTAINER ID        IMAGE               COMMAND             CREATED
+STATUS              PORTS               NAMES
+3cc87ed4205b        5a67a4d66d05        "/bin/bash"         2 hours ago
+Up 2 hours                              competent_feynman
```

In this example, the container name is "competent_feynman" (the name is
random and generated automatically). Using this name, one can then copy
files into the active Docker session from other terminal using:

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
