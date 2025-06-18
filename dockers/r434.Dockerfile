FROM r-base:4.3.4

RUN apt-get update && apt-get install -y \
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
    && mkdir -p /tmp/sandbox_output /tmp/sandbox_plots

# Install essential R packages for data analysis
RUN R -e "install.packages(c('ggplot2', 'dplyr', 'tidyr', 'readr', 'plotly', 'data.table', 'lubridate', 'stringr', 'forcats', 'purrr', 'tibble', 'jsonlite', 'httr', 'devtools'), repos='https://cran.rstudio.com/')"

WORKDIR /sandbox
