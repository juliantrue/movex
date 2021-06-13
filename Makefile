CWD=${PWD}
MOT_PATH=/home/julian/Datasets/MOT20

.PHONY: all
all: build bash


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
			-v ${MOT_PATH}:/MOT \
			-v /home/julian/Datasets/flownet_data:/flownet_data \
			-v ${CWD}:/move \
			move:latest python3 main.py

.PHONY: bash
bash:
		xhost +local:root
		sudo docker run -it --rm \
			--gpus=all \
			-e DISPLAY=unix${DISPLAY} \
			-v /tmp/.X11-unix:/tmp/.X11-unix \
			-v ${MOT_PATH}:/MOT \
			-v /home/julian/Datasets/mot_20_flownet_data:/flownet_data \
			-v ${CWD}:/move \
			move:latest bash

.PHONY: eval
eval:
	python3 parse_metadata.py ./results
	python3 -u -m motmetrics.apps.eval_motchallenge /MOT/train ./results 
	


