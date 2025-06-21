FROM r-base:4.3.4

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
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
RUN R -e "options(repos = c(CRAN = 'https://cran.rstudio.com/')); \
           install.packages( \
             c('ggplot2','dplyr','tidyr','readr','plotly','data.table','lubridate','stringr','forcats','purrr','tibble','jsonlite','httr','devtools'), \
             dependencies = TRUE, \
             quiet = TRUE)"

USER sandbox

WORKDIR /sandbox
