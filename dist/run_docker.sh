#!/bin/bash

docker run -it --rm \
    --entrypoint bash \
    -v $(pwd):/tmp/dist \
    homeassistant/home-assistant /tmp/dist/make_wheel.sh v0.2.38