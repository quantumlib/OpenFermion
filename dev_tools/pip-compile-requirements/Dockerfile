FROM continuumio/miniconda3
WORKDIR /app

RUN conda create -p /py36 python=3.6 && /py36/bin/pip install pip-tools
RUN conda create -p /py37 python=3.7 && /py37/bin/pip install pip-tools
RUN conda create -p /py38 python=3.8 && /py38/bin/pip install pip-tools
RUN conda create -p /py39 python=3.9 && /py39/bin/pip install pip-tools

ADD requirements.in optional-requirements.txt ./
RUN /py36/bin/pip-compile requirements.in optional-requirements.txt --output-file=requirements-py36.txt
RUN /py37/bin/pip-compile requirements.in optional-requirements.txt --output-file=requirements-py37.txt
RUN /py38/bin/pip-compile requirements.in optional-requirements.txt --output-file=requirements-py38.txt
RUN /py39/bin/pip-compile requirements.in optional-requirements.txt --output-file=requirements-py39.txt
