# Docker setup for OpenFermion and select plugins

This Docker image will help users to easily install [OpenFermion](https://github.com/quantumlib/OpenFermion) and its available plugins for [ProjectQ](https://github.com/ProjectQ-Framework/ProjectQ), [Psi4](https://github.com/quantumlib/OpenFermion-Psi4), and [PySCF](https://github.com/quantumlib/OpenFermion-PySCF). Check out Docker's [website](https://www.docker.com/what-container) that describes what a container image is and why it can be so useful.


## What's included?

- Python 2.7 (see dockerfile for instructions on how to change the Python version/distribution)
- Git
- [OpenFermion](https://github.com/quantumlib/OpenFermion)
- [OpenFermion-ProjectQ](https://github.com/quantumlib/OpenFermion-ProjectQ)
- [OpenFermion-Psi4](https://github.com/quantumlib/OpenFermion-Psi4)
- [OpenFermion-PySCF](https://github.com/quantumlib/OpenFermion-PySCF)


## Usage

To use this image, you first need to install [Docker](https://www.docker.com/).
Then, to build the Docker image, move the
[dockerfile](https://github.com/quantumlib/OpenFermion/blob/master/docker/dockerfile)
to your working directory and execute:

```
docker build -t "openfermion_docker" .
```

Finally, to run the image (assuming you're still inside your working directory), execute with `YOUR_WORK_DIR` as the path to your working directory:

```
docker run -it -v $(pwd):YOUR_WORK_DIR -w YOUR_WORK_DIR openfermion_docker
```

When you are done with the Docker image, you can use `docker stop
YOUR_CONTAINER_ID` or `docker kill YOUR_CONTAINER_ID` to stop your container
(you can get your container ID by entering the command `docker ps`). Finally,
feel free to change your copy of the dockerfile to build a more customized
image. For example, you may want to minimze the number of layers by combining the `RUN` statements to further reduce the image size.


## Example

Suppose your working directory were called `openfermion_test`. Go to that
directory (which would contain your scripts) and copy over the dockefile. Build the image. Your run command would then be:

```
docker run -it -v $(pwd):/openfermion_test -w /openfermion_test openfermion_docker
```
