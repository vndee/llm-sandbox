FROM ruby:3.0.2-bullseye

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/sandbox_output /tmp/sandbox_plots /tmp/gem_cache

ENV GEM_HOME=/tmp/gem_cache
ENV PATH=/tmp/gem_cache/bin:$PATH

WORKDIR /sandbox
