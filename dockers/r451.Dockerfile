FROM rocker/r-ver:4.5.1

ENV DEBIAN_FRONTEND=noninteractive

# Fix package repository issues by using more stable approach
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    && apt-get update --fix-missing

# Install development packages with proper error handling
RUN apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r sandbox && useradd -ms /bin/bash -g sandbox sandbox \
    && mkdir -p /tmp/sandbox_output /tmp/sandbox_plots && chown -R sandbox:sandbox /tmp/sandbox_*

# Install essential R packages for data analysis
RUN R -e "install.packages(c('ggplot2', 'dplyr', 'tidyr', 'readr', 'plotly', 'data.table', 'lubridate', 'stringr', 'forcats', 'purrr', 'tibble', 'jsonlite', 'httr', 'devtools'), repos='https://cran.rstudio.com/', dependencies=TRUE)"

USER sandbox

WORKDIR /sandbox
