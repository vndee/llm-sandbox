FROM golang:1.23.4-bullseye

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/sandbox_output /tmp/sandbox_plots /tmp/go_cache

ENV GOCACHE=/tmp/go_cache
ENV GO111MODULE=on

WORKDIR /sandbox
