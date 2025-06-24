#!/bin/bash

docker build -t ghcr.io/vndee/sandbox-r-451-bullseye -f dockers/r451.Dockerfile .
docker push ghcr.io/vndee/sandbox-r-451-bullseye
