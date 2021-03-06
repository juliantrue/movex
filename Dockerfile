FROM tensorflow/tensorflow:latest-gpu

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get -y  --no-install-recommends install software-properties-common
RUN apt-get update && apt-get -y install \
      build-essential \
      libb64-dev \
      libsm6 \
      libxext6 \
      libxrender1 \
      libfontconfig1 \
      libgl1-mesa-glx \
      ffmpeg \
    && apt-get -y autoremove \
    && apt-get clean autoclean


RUN pip3 install --upgrade pip
RUN pip3 install \
      av \
      numpy \
      scipy \
      motmetrics \
      opencv-python==4.4.0.44 \
      opencv-contrib-python==4.4.0.44 \
      tqdm \
      matplotlib


COPY third_party/fig-0.1.0-py3-none-any.whl /tmp/fig-0.1.0-py3-none-any.whl
RUN pip3 install /tmp/fig-0.1.0-py3-none-any.whl


WORKDIR /move
