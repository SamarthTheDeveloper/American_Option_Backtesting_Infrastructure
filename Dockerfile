FROM rust:latest

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        protobuf-compiler \
        libprotobuf-dev && \
    rm -rf /var/lib/apt/lists/* \

COPY . .

CMD ["bash"]