#!/bin/bash
export FORCE_CMAKE=1
export CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --no-cache-dir --upgrade --force-reinstall
pip install -e .
