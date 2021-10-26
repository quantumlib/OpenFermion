#!/bin/bash
python generate-requirements.in.py
docker build -t openfermion .
docker create --name dummy openfermion
docker cp dummy:/app/requirements-py36.txt .
docker cp dummy:/app/requirements-py37.txt .
docker cp dummy:/app/requirements-py38.txt .
docker cp dummy:/app/requirements-py39.txt .
docker rm -f dummy
