#!/bin/bash

echo "[*] Installing Python packages from requirements.txt..."
pip install --upgrade pip
pip install --verbose -r requirements.txt