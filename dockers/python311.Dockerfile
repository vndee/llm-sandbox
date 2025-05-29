FROM python:3.11-bullseye

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /tmp/venv

RUN mkdir -p {/tmp/pip_cache,/tmp/sandbox_output,/tmp/sandbox_plots}

RUN /tmp/venv/bin/pip install --upgrade pip --cache-dir /tmp/pip_cache

RUN /tmp/venv/bin/pip install numpy pandas matplotlib pillow seaborn scikit-learn scipy scikit-image plotly

WORKDIR /sandbox
