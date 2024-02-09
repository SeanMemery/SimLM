#!/bin/bash

# Install python
apt-get update && apt-get install -y python3.11 python3-pip

# Install dependencies
cd /mnt/ceph_rbd/SimLM-3D/pooltool-text
python3 -m pip install -e .
python3 -m pip uninstall panda3d -y
python3 -m pip install --pre --extra-index-url https://archive.panda3d.org/ panda3d
cd ..
python3 -m pip install tqdm jinja2 requests

# Start evaluation
python3 evaluate_model.py