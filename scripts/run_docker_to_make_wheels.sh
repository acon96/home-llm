#!/bin/bash

VERSION_TO_BUILD="v0.2.77"

# make python11 wheels
# docker run -it --rm \
#     --entrypoint bash \
#     -v $(pwd):/tmp/dist \
#     homeassistant/home-assistant:2023.12.4 /tmp/dist/make_wheel.sh $VERSION_TO_BUILD

# make python 12 wheels
docker run -it --rm \
    --entrypoint bash \
    -v $(pwd):/tmp/dist \
    homeassistant/home-assistant:2024.2.1 /tmp/dist/make_wheel.sh $VERSION_TO_BUILD