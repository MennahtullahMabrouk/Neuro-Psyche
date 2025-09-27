#!/usr/bin/env bash

echo "Configuring local environment for development..."
if [[ -d ./venv ]] ; then
    rm -rf ./venv
fi

if ! [[ -d ./venv ]] ; then
    python3 -m venv ./venv
fi

echo "Activating virtual environment..."
source ./venv/bin/activate

echo "Installing packages..."
python3 -m pip install --upgrade pip
python3 -m pip install poetry
poetry lock --no-cache
poetry install --no-root

echo "Ready for development!"