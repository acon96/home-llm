#!/bin/bash

docker run -it \
    --entrypoint bash \
    -v $(pwd):/tmp/dist \
    homeassistant/home-assistant /tmp/dist/make_wheel.sh