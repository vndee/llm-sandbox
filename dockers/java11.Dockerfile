FROM openjdk:11.0.12-jdk-bullseye

RUN apt-get update && apt-get install -y \
    build-essential \
    maven \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/sandbox_output /tmp/sandbox_plots /tmp/maven_cache

ENV MAVEN_OPTS="-Dmaven.repo.local=/tmp/maven_cache"

WORKDIR /sandbox
