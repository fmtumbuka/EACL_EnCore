#!/usr/bin/env bash


export PYTHONPATH=`pwd`/src/main/python:${PYTHONPATH}
python3 -m unittest discover -s src/test/python -p "*_test.py"