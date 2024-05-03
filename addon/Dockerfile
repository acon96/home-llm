ARG BUILD_FROM=ghcr.io/hassio-addons/ubuntu-base:9.0.2

# hadolint ignore=DL3006
FROM ${BUILD_FROM}

# Set shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install text-generation-webui
ARG BUILD_ARCH=amd64
ARG APP_DIR=/app
RUN \
    apt-get update \
    \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
        cmake \
        python3.10 \
        python3-dev \
        python3-venv \
        python3-pip \
    \
    && git clone https://github.com/oobabooga/text-generation-webui.git ${APP_DIR} --branch snapshot-2024-04-28 \
    && python3 -m pip install torch torchvision torchaudio py-cpuinfo==9.0.0 \
    && python3 -m pip install -r ${APP_DIR}/requirements_cpu_only_noavx2.txt llama-cpp-python \
    && apt-get purge -y --auto-remove \
        git \
        build-essential \
        cmake \
        python3-dev \
    && apt-get clean \
    && rm -fr \
        /tmp/* \
        /var/{cache,log}/* \
        /var/lib/apt/lists/*

# Copy root filesystem for our image
COPY rootfs /

# Build arugments
ARG BUILD_DATE
ARG BUILD_REF
ARG BUILD_VERSION
ARG BUILD_REPOSITORY

# Labels
LABEL \
    io.hass.name="oobabooga text-generation-webui for ${BUILD_ARCH}" \
    io.hass.description="Home Assistant Community Add-on: ${BUILD_ARCH} oobabooga text-generation-webui" \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.type="addon" \
    io.hass.version=${BUILD_VERSION} \
    maintainer="github.com/acon96" \
    org.opencontainers.image.title="oobabooga text-generation-webui for ${BUILD_ARCH}" \
    org.opencontainers.image.description="Home Assistant Community Add-on: ${BUILD_ARCH} oobabooga text-generation-webui" \
    org.opencontainers.image.vendor="Home Assistant Community Add-ons" \
    org.opencontainers.image.authors="github.com/acon96" \
    org.opencontainers.image.licenses="TBD" \
    org.opencontainers.image.url="https://addons.community" \
    org.opencontainers.image.source="https://github.com/${BUILD_REPOSITORY}" \
    org.opencontainers.image.documentation="https://github.com/${BUILD_REPOSITORY}/blob/main/README.md" \
    org.opencontainers.image.created=${BUILD_DATE} \
    org.opencontainers.image.revision=${BUILD_REF} \
    org.opencontainers.image.version=${BUILD_VERSION}
