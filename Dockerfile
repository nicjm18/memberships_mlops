FROM jenkins/jenkins:lts-jdk17

USER root

# Instalar Python y dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Crear symlink para python
RUN ln -s /usr/bin/python3 /usr/bin/python

USER jenkins