#!/bin/sh

# You may use this initialization script to easily setup an Onyxia "vscode-python" service
# https://datalab.sspcloud.fr/launcher/ide/vscode-python?name=synth-data&init.personalInit=%C2%ABhttps%3A%2F%2Fraw.githubusercontent.com%2FInseeFrLab%2Fdata-reconstructio-from-tiles%2Frefs%2Fheads%2Fmain%2Finit-scripts%2Fvscode-python.sh%C2%BB

sudo apt update -y
sudo apt install tree -y

# Clone project
git clone https://github.com/InseeFrLab/diff-privacy-poc.git
cd diff-privacy-poc

# Install project (requirements and main package)
pip install -e .

# Replace default flake8 linter with project-preconfigured ruff
code-server --uninstall-extension ms-python.flake8
code-server --install-extension charliermarsh.ruff

# Install mypy and type stubs + install the mypy type checking extension
pip install mypy
yes | mypy --install-types
code-server --install-extension ms-python.mypy-type-checker
