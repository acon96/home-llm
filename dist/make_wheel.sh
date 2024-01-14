#!/bin/bash
# Don't run this. This is executed inside of the home assistant container to build the wheel

apk update
apk add build-base python3-dev

cd /tmp
git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python --branch $1
cd llama-cpp-python
pip3 install build
python3 -m build --wheel
cp -f ./dist/*.whl /tmp/dist/
