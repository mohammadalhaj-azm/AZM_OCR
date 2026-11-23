#!/usr/bin/env bash
# Prepare an Ubuntu 22.04 GPU host (e.g. AWS g6) for docker-compose.gpu.yml: NVIDIA driver, Docker Engine with the
# compose plugin, and the NVIDIA Container Toolkit. Each step is skipped when it is already in place (the AWS
# "Deep Learning Base ... GPU AMI" ships most of it), so the script can be run again safely.
#
#   bash scripts/ec2_setup.sh
#
# If it installs the driver it asks for a reboot; run it again afterwards to finish.
set -euo pipefail

say() { printf '\n==> %s\n' "$*"; }

if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi
. /etc/os-release
if [ "${ID:-}" != "ubuntu" ]; then
    echo "This script is for Ubuntu (found: ${PRETTY_NAME:-unknown})." >&2
    exit 1
fi
[ "${VERSION_ID:-}" = "22.04" ] || echo "Note: written for Ubuntu 22.04, found ${VERSION_ID:-unknown}; continuing."

# 1. NVIDIA driver
say "NVIDIA driver"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    $SUDO apt-get update
    $SUDO apt-get install -y ubuntu-drivers-common
    $SUDO ubuntu-drivers install
    echo
    echo "Driver installed. Reboot (sudo reboot), reconnect, and run this script again."
    exit 0
fi

# 2. Docker Engine + compose plugin
say "Docker"
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    docker --version
    docker compose version
else
    curl -fsSL https://get.docker.com | $SUDO sh
fi
if [ -n "$SUDO" ] && ! id -nG "$USER" | grep -qw docker; then
    $SUDO usermod -aG docker "$USER"
    ADDED_TO_GROUP=1
fi

# 3. NVIDIA Container Toolkit (lets containers use the GPU)
say "NVIDIA Container Toolkit"
if command -v nvidia-ctk >/dev/null 2>&1; then
    nvidia-ctk --version | head -1
else
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | $SUDO gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    $SUDO apt-get update
    $SUDO apt-get install -y nvidia-container-toolkit
fi
$SUDO nvidia-ctk runtime configure --runtime=docker >/dev/null
$SUDO systemctl restart docker

# 4. Check: a container sees the GPU
say "GPU inside a container"
$SUDO docker run --rm --gpus all ubuntu:22.04 nvidia-smi

say "Host ready."
if [ "${ADDED_TO_GROUP:-0}" = 1 ]; then
    echo "You were added to the docker group: log out and back in (or run 'newgrp docker') before using docker without sudo."
fi
