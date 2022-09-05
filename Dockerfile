FROM python:3.6.15-slim-buster

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-dev tzdata gosu && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    pip install pipenv==2022.4.8

WORKDIR /app

COPY Pipfile Pipfile.lock entrypoint.sh /app/

ENTRYPOINT [ "/app/entrypoint.sh" ]
