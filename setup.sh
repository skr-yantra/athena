#!/usr/bin/env bash

# Upgrade ubuntu
sudo apt update
sudo apt upgrade -y

# Install CUDA
bash setup_cuda.sh

# Install build deps
sudo apt install -y build-essential libpython3-dev python3-venv

# Create virtual env
python3 -m venv ~/.env

# Auto source to .env
echo "source ~/.env/bin/activate" >> ~/.bashrc

# Install python dependencies
source ~/.env/bin/activate
pip install --upgrade pip
pip install -r requirements_gpu.txt
pip install -r requirements.txt

# Reboot
sudo reboot