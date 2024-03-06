#!/bin/bash

set -e
set -x

python -m venv venv_d3pm
source venv_d3pm/bin/activate
pip install --upgrade pip
pip install -r requirements.txt