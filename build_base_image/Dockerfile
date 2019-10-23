FROM alpine:edge

COPY . /app

WORKDIR /app

RUN apk update && apk add build-base libzmq musl-dev python3 python3-dev zeromq-dev

RUN pip3 install -r requirements.txt

# reduce image size by cleaning up the build packages
RUN apk del build-base musl-dev python3-dev zeromq-dev
