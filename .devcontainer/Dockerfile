FROM mcr.microsoft.com/devcontainers/base:bullseye

USER vscode

# Install uv
RUN mkdir ~/build && cd ~/build && \
    wget https://astral.sh/uv/install.sh && \
    chmod +x ./install.sh && ./install.sh && \
    rm -rf ~/build
ENV PATH="/home/vscode/.local/bin/:$PATH"
