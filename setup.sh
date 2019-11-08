#!/usr/bin/env bash

# Upgrade ubuntu
sudo apt update
sudo apt upgrade -y

# Install CUDA
bash setup_cuda.sh

# Install build deps
sudo apt install -y libpython3-dev python3-venv

# Create virtual env
python3 -m venv ~/.env

# Auto source to .env
echo "source ~/.env/bin/activate" >> ~/.bashrc

# Reboot
sudo reboot