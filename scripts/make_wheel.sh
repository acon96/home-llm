#!/bin/bash
# Don't run this. This is executed inside of the home assistant container to build the wheel

apk update
apk add build-base python3-dev

cd /tmp
git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python --branch $1
cd llama-cpp-python
pip3 install build

tag="homellm"
sed -i -E "s/^(__version__ *= *\"[0-9]+\.[0-9]+\.[0-9]+)\"/\1+${tag}\"/" llama_cpp/__init__.py

export CMAKE_ARGS="-DLLAVA_BUILD=OFF -DGGML_NATIVE=ON"
python3 -m build --wheel
cp -f ./dist/*.whl /tmp/dist/
