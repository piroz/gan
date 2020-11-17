FROM tensorflow/tensorflow:1.12.3-gpu-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-dev tzdata gosu && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install pipenv

WORKDIR /app

COPY Pipfile Pipfile.lock entrypoint.sh /app/
