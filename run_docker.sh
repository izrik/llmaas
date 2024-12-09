#!/bin/bash

# TODO replace "$PWD" with "$BASH_SOURCE" etc
docker run --rm -it -v "$PWD:/host" -v "$PWD/models:/llmaas/models" -p 5000:5000 --name llmaas "$@" llmaas
