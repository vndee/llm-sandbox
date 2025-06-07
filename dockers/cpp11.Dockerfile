FROM gcc:11.2.0-bullseye

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/sandbox_output /tmp/sandbox_plots

WORKDIR /sandbox
