# Docker setup for OpenFermion and select plugins

This Docker image will help users to easily install [OpenFermion](https://github.com/quantumlib/OpenFermion) and its available plugins for [ProjectQ](https://github.com/ProjectQ-Framework/ProjectQ), [Psi4](https://github.com/quantumlib/OpenFermion-Psi4), and [PySCF](https://github.com/quantumlib/OpenFermion-PySCF). Check out Docker's [website](https://www.docker.com/what-container) that describes what a container image is and why it can be so useful.


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


## Usage

To use this image, you first need to install [Docker](https://www.docker.com/).
Then, to build the Docker image, move the
[dockerfile](https://github.com/quantumlib/OpenFermion/blob/master/docker/dockerfile)
to your working directory and execute:

```
docker build -f dockerfile -t openfermion_docker .
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


## Setting up Docker for the first time


When Docker is installed, open a command line terminal and check the list of
running virtual machines by

```
	docker-machine ls
```

The returned list should be empty if this is the first time Docker is run on
the computer. Create a virtual machine by running

```
	docker-machine create --driver virtualbox default
```

(Note: To be able to run this, one needs to install [virtualbox](https://www.virtualbox.org/wiki/Downloads))

Here "default" is just the name of the virtual machine. You can replace it by
any name that you prefer. To check that the virtual machine is running indeed,
use `docker-machine ls` again.

When the Docker virtual machine is created, configure the shell by running

```
	docker-machine env default
```

where if you named the virtual machine differently from default you should also
replace "default" with the customized name. The command above will return a
message containing the command to run for configuring the shell. This command
depends on the OS.

3. Run the command in the message returned above.


## Running OpenFermion with Docker
-------------------------------

Now that Docker is set up, one could navigate to the folder containing the
Dockerfile for building the OpenFermion image (docker/dockerfile) and run

```
	docker build -t openfermion_docker .
```

where "openfermion_docker" is just an arbitrary name and one could replace it
with any name she deems sensible. Here we will use kickass_openfermion as an
example.

It takes a few minutes to build the image. What the Dockerfile does is to
start from a base image of ubuntu and install OpenFermion, its plugins and the
necessary applications needed for running these programs. To run the image, use

```
	docker run -it openfermion_docker
```

and the terminal enters a new environment which emulates a Ubuntu OS with
OpenFermion and accessories installed, regardless of what the host OS is. This
new environment is a running process called a Docker container. To check info
on the container, one can open another terminal, configure it using step 3, and
run

```
	docker ps
```

which returns a list of running containers. For example it might look like:

```
CONTAINER ID        IMAGE               COMMAND             CREATED             
STATUS              PORTS               NAMES
3cc87ed4205b        5a67a4d66d05        "/bin/bash"         29 hours ago        
Up 29 hours                             blissful_brown
```

in which case we have a running container called blissful_brown that was
started quite a while ago.

The freshly built image is ready to run any Python program that uses
OpenFermion. To transfer files from somewhere on the disk to the Docker
container, run in a separate terminal from the one running the container

```
	docker cp [path to file on disk] [container name]:[path in container]
```

where container name can be gleaned according to step 6 above.

An alternative way of loading files onto the Docker container is through
remote repos such as Github or BitBucket. git is installed in the Docker image
built in step 5. After step 6, one could run "git clone ..." etc to pull files
remotely into the Docker container.

There are occasions where one might want to open up multiple terminals to
run the same Docker container. In that case, one could run in any terminal

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

Assuming the Docker image for OpenFermion is already built and as an 
example we assume it is called kickass_openfermion, run the container with an
additional -p flag:

```
	docker run -it -p 8888:8888 kickass_openfermion
```

Here the numbers 8888 simply specifies the port number through which the Docker
container communicates with the browser. If for some reason this port is not
available, any other number in 8000-9000 will do.

When the terminal enters the Docker container, run Jupyter notebook by

```
	jupyter-notebook --allow-root --no-browser --port 8888 --ip=0.0.0.0
```

where 8888 is the port number used in step 11 for setting up the container.
The message returned to the terminal may look something like

```
[I 21:03:12.979 NotebookApp] Writing notebook server cookie secret to /root/.loc
al/share/jupyter/runtime/notebook_cookie_secret
[I 21:03:13.001 NotebookApp] Serving notebooks from local directory: /
[I 21:03:13.001 NotebookApp] 0 active kernels
[I 21:03:13.002 NotebookApp] The Jupyter Notebook is running at:
[I 21:03:13.002 NotebookApp] http://0.0.0.0:8888/?token=8f70c035fb9b0dbbf160d996
f7f341fecf94c9aedc7cfaf7
[I 21:03:13.002 NotebookApp] Use Control-C to stop this server and shut down all
 kernels (twice to skip confirmation).
[C 21:03:13.002 NotebookApp] 
    
Copy/paste this URL into your browser when you connect for the first time,
to login with a token:
   http://0.0.0.0:8888/?token=8f70c035fb9b0dbbf160d996f7f341fecf94c9aedc7cfaf7
```

Note the token string 8f70c035fb9b0dbbf160d996f7f341fecf94c9aedc7cfaf7.

Open a browser window and type in the address line

```
	[virtual machine ip]:8888
```

where [virtual machine ip] is extracted from step 10 and 8888 is the port 
number (or any other port number that one specifies in step 11). A webpage
asking for token string should appear. Use the token string in step 12 to
enter Jupyter Notebook.

If logged in successfully, you should be able to freely navigate through
the entire Docker image and launch any Jupyter notebook in the image.
