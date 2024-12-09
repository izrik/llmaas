#!/bin/bash

VERSION=$1
if [[ -z "$VERSION" ]]; then
  VERSION=$(git describe --tags | sed 's/^v//' )
fi
if [[ -z "$VERSION" ]]; then
  echo "Error: Could not determine the version string." >&2
  exit 1
fi

echo "Building version $VERSION..."
docker build -t llmaas:latest --build-arg "VERSION=$VERSION" .

docker tag llmaas:latest "llmaas:$VERSION"
docker tag llmaas:latest "registry.re.metaindu.com/llmaas:$VERSION"
