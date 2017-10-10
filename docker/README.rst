Docker Setup for FermiLib + ProjectQ
====================================

This Docker image will help users to easily install `FermiLib <https://github.com/ProjectQ-Framework/FermiLib.git>`__ and `ProjectQ <https://github.com/ProjectQ-Framework/ProjectQ>`__. Check out Docker's `website <https://www.docker.com/what-container>`__ that describes what a container image is and why it can be so useful.

What is included?
-----------------

- Python 2.7 (you can also use Python 3 with one minor change in the Dockerfile. See the Dockerfile for instructions.)
- `ProjectQ <https://github.com/ProjectQ-Framework/ProjectQ.git>`__ 
- `FermiLib <https://github.com/ProjectQ-Framework/FermiLib.git>`__

How to use it?
--------------

1. To use this image, you first need to install `Docker <https://www.docker.com/>`__.

2. To build the Docker image, move the Dockerfile to your working directory. Then execute:

.. code-block:: bash

        docker build -t "fermiq_docker" .

3. To run the image (assuming you're still inside your working directory), execute with :code:`YOUR_WORK_DIR` as the path to your working directory:

.. code-block:: bash

        docker run -it -v $(pwd):YOUR_WORK_DIR -w YOUR_WORK_DIR fermiq_docker

When you're done using the Docker image, you can execute :code:`docker stop YOUR_CONTAINER_ID` or :code:`docker kill YOUR_CONTAINER_ID` to stop your container (you can get the container ID by using the command :code:`docker ps`). Finally, feel free to use this as a parent image to build a more customized image layer, perhaps containing the available plugins (`PySCF <https://github.com/ProjectQ-Framework/FermiLib-Plugin-PySCF>`__ or `Psi4 <https://github.com/ProjectQ-Framework/FermiLib-Plugin-Psi4>`__) for FermiLib.
