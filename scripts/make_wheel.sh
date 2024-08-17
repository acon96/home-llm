#!/bin/bash
# Don't run this. This is executed inside of the home assistant container to build the wheel

apk update
apk add build-base python3-dev cmake

cd /tmp
git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python --branch $1
cd llama-cpp-python
pip3 install build

# for some reason, scikit-build-core v0.9.7+ doesn't produce properly tagged musllinux wheels
sed -i -e 's/scikit-build-core\[pyproject\]>=0.9.2/scikit-build-core\[pyproject\]==0.9.6/g' pyproject.toml

python3 -m build --wheel
cp -f ./dist/*.whl /tmp/dist/
