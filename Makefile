CWD=${PWD}
MOT16_PATH=/home/julian/Datasets/MOT16

.PHONY: all
all: build run


.PHONY: build
build:
		sudo docker build -t move:latest .

.PHONY: run
run:
		xhost +local:root
		sudo docker run -it --rm \
			--gpus=all \
			-e DISPLAY=unix${DISPLAY} \
			-v /tmp/.X11-unix:/tmp/.X11-unix \
			-v ${MOT16_PATH}:/MOT16 \
			-v ${CWD}:/move \
			move:latest bash

.PHONY: eval
eval:
	python3 -u -m motmetrics.apps.eval_motchallenge /MOT16/train ./results/ --fmt mot16
