#!/bin/bash

VERSION_TO_BUILD="0.3.20"

# Build generic py3-none wheel (works for all Python 3.x versions)
docker run -it --rm \
    --entrypoint bash \
    -v $(pwd):/tmp/dist \
    homeassistant/home-assistant:2025.4.1 /tmp/dist/make_wheel.sh $VERSION_TO_BUILD