#!/bin/bash

MODEL_NAME=${1}
REMOTE_SERVER=${2}

if [ -z "$MODEL_NAME" ] || [ -z "$REMOTE_SERVER" ]; then
  echo "Usage: $0 <config-name> <remote-server>"
  exit 1
fi

scp configs/${MODEL_NAME}.yml ${REMOTE_SERVER}:/mnt/data/training-configs/
cat training-job.yml | sed "s/MODEL_NAME/${MODEL_NAME}/g" | kubectl create -f -
