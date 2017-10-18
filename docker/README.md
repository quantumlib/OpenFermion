# Docker setup for OpenFermion and select plugins

This Docker image will help users to install [OpenFermion](https://github.com/quantumlib/OpenFermion) and its available plugins for [ProjectQ](https://github.com/ProjectQ-Framework/ProjectQ), [Psi4](https://github.com/quantumlib/OpenFermion-Psi4), and [PySCF](https://github.com/quantumlib/OpenFermion-PySCF).
Check out Docker's [website](https://www.docker.com/what-container) that describes what a container image is and why it can be so useful.
The Docker based installation is extremely robust and runs on any operating
system and so is an ideal solution for anyone having difficulty installing
OpenFermion (or any of its plugins) using the standard installation.


## What's included?

- Python 2.7 (see dockerfile for instructions on how to change the Python version/distribution)
- Git
- [OpenFermion](https://github.com/quantumlib/OpenFermion)
- [OpenFermion-ProjectQ](https://github.com/quantumlib/OpenFermion-ProjectQ)
- [OpenFermion-Psi4](https://github.com/quantumlib/OpenFermion-Psi4)
- [OpenFermion-PySCF](https://github.com/quantumlib/OpenFermion-PySCF)
- [ProjectQ](https://projectq.ch)
- [Psi4](http://www.psicode.org)
- [PySCF](https://github.com/sunqm/pyscf)


## Setting up Docker for the first time

You first need to install [Docker](https://www.docker.com/).
Once Docker is installed, open a command line terminal and check the list of
running virtual machines with

```
docker-machine ls
```

Assuming this is the firt time Docker has been run, the list should be empty.
Create a virtual machine by running

```
docker-machine create --driver virtualbox default
```

To be able to run this, one needs to install
[VirtualBox](https://www.virtualbox.org/wiki/Downloads).
Here, "default" is just the name of the virtual machine. You can replace it by
any name that you prefer. To check that the virtual machine is indeed running,
use `docker-machine ls` again.
After the Docker virtual machine is created, configure the shell by running

```
docker-machine env default
```

where if you named the virtual machine differently from default you should also
replace "default" with the customized name. The command above will return an OS
dependent message containing the command to run for configuring the shell;
follow those instructions.


## Running OpenFermion with Docker

Now that Docker is set up, one can navigate to the folder containing the
Dockerfile for building the OpenFermion image (docker/dockerfile) and run

```
docker build -t openfermion_docker .
```

where "openfermion_docker" is just an arbitrary name for our docker container.
What the Dockerfile does is to start from a base image of Ubuntu and install
OpenFermion, its plugins and the necessary applications needed for running these
programs. This is a fairly involved installation and will take some time 
(perhaps up to thiry minutes depending on the computer). Once installation has
completed, run the image with

```
docker run -it openfermion_docker
```

With this command the terminal enters a new environment which emulates a Ubuntu OS with
OpenFermion and accessories installed. The freshly built image is ready to run any Python program that uses
OpenFermion. To transfer files from somewhere on the disk to the Docker
container, run in a separate terminal from the one running the container

```
docker cp [path to file on disk] [container name]:[path in container]
```

where container name is "openfermion_docker" in this case.

An alternative way of loading files onto the Docker container is through
remote repos such as Github or BitBucket. git is installed in the Docker image.
After `docker run`, one could run "git clone ..." etc to pull files
remotely into the Docker container. There are occasions where one might want
to open up multiple terminals to run the same Docker container.
In that case, one could run in any terminal

```
docker exec -it [container name] bash
```

and "get into" the container.

## Running Jupyter notebook with Docker backend

To run Jupyter notebook in a browser with a Docker container running as a
backend, first check the ip address of the virtual machine by running

```
docker-machine ip default
```

where "default" can be replaced by the name of whichever virtual machine whose
ip address you want to check.

Assuming the Docker image for OpenFermion is already built and is called
openfermion_docker, run the container with an additional -p flag:


```
docker run -it -p 8888:8888 openfermion_docker
```

Here the numbers 8888 simply specifies the port number through which the Docker
container communicates with the browser. If for some reason this port is not
available, any other number in 8000-9000 will do.

When the terminal enters the Docker container, run Jupyter notebook by

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
enter Jupyter Notebook. If logged in successfully, you should be able to freely
navigate through the entire Docker image and launch any Jupyter notebook in the image.
