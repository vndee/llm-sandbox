FROM rocker/r-ver:4.5.1

ENV DEBIAN_FRONTEND=noninteractive

# Fix package repository issues by using more stable approach
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    && apt-get update --fix-missing

# Install only essential development packages for core data science
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

# Install parallel package management tool
RUN R -e "install.packages('pak', repos='https://cran.rstudio.com/', dependencies=TRUE)"

# Install all R packages in a single parallelized step using pak
RUN R -e "pak::pkg_install(c( \
  # Core tidyverse and data manipulation packages
  'tidyverse', 'data.table', \
  \
  # Essential visualization packages
  'plotly', 'gridExtra', 'RColorBrewer', 'viridis', \
  \
  # Core statistical modeling and machine learning packages
  'caret', 'randomForest', 'glmnet', 'broom', \
  \
  # Time series analysis packages
  'forecast', 'zoo', 'xts', 'lubridate', \
  \
  # Basic text analysis packages
  'tm', 'stringr', \
  \
  # Essential utility packages
  'devtools', 'here', 'janitor', \
  \
  # BiocManager for Bioconductor packages
  'BiocManager', \
  \
  # Core Bioconductor packages for biological analysis
  'Biobase', 'BiocGenerics', 'S4Vectors', 'IRanges', 'GenomeInfoDb', \
  \
  # Essential genomics and bioinformatics packages
  'GenomicRanges', 'GenomicFeatures', 'AnnotationDbi', 'org.Hs.eg.db', \
  \
  # Popular analysis packages
  'DESeq2', 'edgeR', 'limma', 'ComplexHeatmap', 'EnhancedVolcano', \
  \
  # Additional useful Bioconductor packages
  'biomaRt', 'GO.db', 'KEGGREST', 'clusterProfiler' \
), dependencies=TRUE, upgrade=FALSE)"

USER sandbox

WORKDIR /sandbox
