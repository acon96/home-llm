#!/bin/bash
docker run -it --rm \
  --name homeassistant \
  --privileged \
  -e TZ=MY_TIME_ZONE \
  -v $(pwd)/config:/config \
  --network=host \
  ghcr.io/home-assistant/home-assistant:stable