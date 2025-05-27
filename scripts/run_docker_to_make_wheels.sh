#!/bin/bash

VERSION_TO_BUILD="v0.3.9"

# make python 11 wheels
# docker run -it --rm \
#     --entrypoint bash \
#     -v $(pwd):/tmp/dist \
#     homeassistant/home-assistant:2023.12.4 /tmp/dist/make_wheel.sh $VERSION_TO_BUILD

# make python 12 wheels
# docker run -it --rm \
#     --entrypoint bash \
#     -v $(pwd):/tmp/dist \
#     homeassistant/home-assistant:2024.2.1 /tmp/dist/make_wheel.sh $VERSION_TO_BUILD

# make python 13 wheels
docker run -it --rm \
    --entrypoint bash \
    -v $(pwd):/tmp/dist \
    homeassistant/home-assistant:2025.4.1 /tmp/dist/make_wheel.sh $VERSION_TO_BUILD