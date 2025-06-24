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

# Install core tidyverse and data manipulation packages
RUN R -e "install.packages(c('tidyverse', 'data.table'), repos='https://cran.rstudio.com/', dependencies=TRUE)"

# Install essential visualization packages
RUN R -e "install.packages(c('plotly', 'gridExtra', 'RColorBrewer', 'viridis'), repos='https://cran.rstudio.com/', dependencies=TRUE)"

# Install core statistical modeling and machine learning packages
RUN R -e "install.packages(c('caret', 'randomForest', 'glmnet', 'broom'), repos='https://cran.rstudio.com/', dependencies=TRUE)"

# Install time series analysis packages
RUN R -e "install.packages(c('forecast', 'zoo', 'xts', 'lubridate'), repos='https://cran.rstudio.com/', dependencies=TRUE)"

# Install basic text analysis packages
RUN R -e "install.packages(c('tm', 'stringr'), repos='https://cran.rstudio.com/', dependencies=TRUE)"

# Install essential utility packages
RUN R -e "install.packages(c('devtools', 'here', 'janitor'), repos='https://cran.rstudio.com/', dependencies=TRUE)"

# Install BiocManager for Bioconductor packages
RUN R -e "install.packages('BiocManager', repos='https://cran.rstudio.com/')"

# Install core Bioconductor packages for biological analysis
RUN R -e "BiocManager::install(c('Biobase', 'BiocGenerics', 'S4Vectors', 'IRanges', 'GenomeInfoDb'), ask = FALSE, update = FALSE)"

# Install essential genomics and bioinformatics packages
RUN R -e "BiocManager::install(c('GenomicRanges', 'GenomicFeatures', 'AnnotationDbi', 'org.Hs.eg.db'), ask = FALSE, update = FALSE)"

# Install popular analysis packages
RUN R -e "BiocManager::install(c('DESeq2', 'edgeR', 'limma', 'ComplexHeatmap', 'EnhancedVolcano'), ask = FALSE, update = FALSE)"

# Install additional useful Bioconductor packages
RUN R -e "BiocManager::install(c('biomaRt', 'GO.db', 'KEGGREST', 'clusterProfiler'), ask = FALSE, update = FALSE)"
USER sandbox

WORKDIR /sandbox
