#!/bin/bash

VERSION=$1
shift
if [[ -z "$VERSION" || "$VERSION" == '-' || "$VERSION" == '--' ]]; then
  VERSION=$(git describe --tags | sed 's/^v//' )
fi
if [[ -z "$VERSION" ]]; then
  echo "Error: Could not determine the version string." >&2
  exit 1
fi

echo "Running version $VERSION..."

# TODO replace "$PWD" with "$BASH_SOURCE" etc
docker run --rm -it -v "$PWD:/host" -v "$PWD/models:/llmaas/models" -p 5000:5000 --name llmaas "$@" "llmaas:$VERSION"
