import sys
import os
import subprocess

import numpy as np


def main():
    mot_path = "/MOT"
    target_folder_path = "data"
    for top_level_folder in ["train", "test"]:
        top_level_folder_path = os.path.join(mot_path, top_level_folder)
        for trace in os.listdir(top_level_folder_path):
            trace_path = os.path.join(top_level_folder_path, trace)
            target_trace_folder_path = os.path.join(target_folder_path, trace)

            create_folder_if_not_exists(target_trace_folder_path)
            convert_mot_trace(trace_path, target_trace_folder_path)


def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def convert_mot_trace(path_to_mot_trace, target_folder_path):

    metadata_ini_path = os.path.join(path_to_mot_trace, "seqinfo.ini")
    metadata = extract_metadata(metadata_ini_path)
    images_path = os.path.join(path_to_mot_trace, "img1")
    target_video_path = os.path.join(
        target_folder_path, os.path.split(target_folder_path)[-1] + ".mp4"
    )
    images_to_video(images_path, target_video_path, metadata)

    detections_path = os.path.join(path_to_mot_trace, "det", "det.txt")
    target_detections_path = os.path.join(
        target_folder_path, os.path.split(target_folder_path)[-1] + ".npy"
    )
    mot_detections_to_npy(detections_path, target_detections_path)


def extract_metadata(path_to_metadata_file):
    if not os.path.exists(path_to_metadata_file):
        print(f"Metadata file at {path_to_metadata_file} does not exist.")
        print("Exiting.")
        sys.exit(1)

    with open(path_to_metadata_file, "r") as f:
        data = f.readlines()

    data = data[1:-1]
    metadata = {kv[0]: kv[1].strip() for kv in map(lambda x: x.split("="), data)}
    return metadata


def images_to_video(path_to_image_folder, video_out_path, metadata):
    fps = int(metadata["frameRate"])
    name = metadata["name"]
    width = metadata["imWidth"]
    height = metadata["imHeight"]

    images_path = os.path.join(path_to_image_folder, "%06d.jpg")
    command = " ".join(
        [
            "ffmpeg",
            "-y",
            f"-r {fps}",
            f"-i {images_path}",
            f"-vf scale=-2:{height}",
            "-vcodec libx264",
            "-preset slow",
            "-pix_fmt yuv420p",
            f"{video_out_path}",
        ]
    )
    print(f"Running {command}")
    subprocess.run(command.split(" "))


def mot_detections_to_npy(path_to_detections_file, path_to_target_detections_file):
    arr = np.loadtxt(path_to_detections_file, delimiter=",")
    np.save(path_to_target_detections_file, np.asarray(arr), allow_pickle=False)
    print(f"Detections saved to {path_to_target_detections_file}")


if __name__ == "__main__":
    main()
