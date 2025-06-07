FROM node:22-bullseye

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/sandbox_output /tmp/sandbox_plots /tmp/npm_cache

ENV NPM_CONFIG_CACHE=/tmp/npm_cache

WORKDIR /sandbox
