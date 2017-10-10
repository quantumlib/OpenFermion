# Docker Setup for OpenFermion and its plugins (ProjectQ, Psi4, PySCF)

This Docker image will help users to easily install [OpenFermion](https://github.com/quantumlib/OpenFermion) and its available plugins for [ProjectQ](https://github.com/ProjectQ-Framework/ProjectQ), [Psi4](https://github.com/quantumlib/OpenFermion-Psi4), and [PySCF](https://github.com/quantumlib/OpenFermion-PySCF). Check out Docker's [website](https://www.docker.com/what-container) that describes what a container image is and why it can be so useful.


## What is included?

- Python 2.7 (Please see Dockerfile for instructions on how to change the Python version/distribution.)
- Git
- [OpenFermion-ProjectQ](https://github.com/quantumlib/OpenFermion-ProjectQ) (Plugin that installs both OpenFermion and ProjectQ)
- [OpenFermion-Psi4](https://github.com/quantumlib/OpenFermion-Psi4)
- [OpenFermion-PySCF](https://github.com/quantumlib/OpenFermion-PySCF)


## Usage

To use this image, you first need to install [Docker](https://www.docker.com/).

After installation, to build the Docker image, move the [Dockerfile]() to your working directory. Then execute:

```
docker build -t "openfermion_docker" .
```

Finally, to run the image (assuming you're still inside your working directory), execute with `YOUR_WORK_DIR` as the path to your working directory:

```
docker run -it -v $(pwd):YOUR_WORK_DIR -w YOUR_WORK_DIR openfermion_docker
```

When you are done using the Docker image, you can use `docker stop YOUR_CONTAINER_ID` or `docker kill YOUR_CONTAINER_ID` to stop your container (you can get your container ID by entering the command `docker ps`). Finally, feel free to change your copy of the Dockerfile to build a more customized image. For example, for your own use, you may want to minimze the number of layers by combining the `RUN` statements to further reduce the image size.


## Example

Suppose your working directory was called `openfermion_test`. Go to that directory (which would contain your scripts) and copy over the Dockefile. Build the image. Your run command would then be:

```
docker run -it -v $(pwd):/openfermion_test -w /openfermion_test openfermion_docker
```
