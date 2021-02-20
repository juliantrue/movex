import os
import pickle
import cv2


def main():
    path_to_result_metadata_file = "results/MOT16-04.metadata"
    path_to_mot_dir = "data"

    metadata = parse_result_metadata_file(path_to_result_metadata_file)
    # mot_trace_name = metadata["trace_name"]
    mot_trace_name = "MOT16-04"
    path_to_video_file = os.path.join(
        path_to_mot_dir, mot_trace_name, mot_trace_name + ".mp4"
    )

    video = video_generator(path_to_video_file)
    render_video_with_metadata(video, metadata)


def parse_result_metadata_file(path_to_result_metadata_file):
    with open(path_to_result_metadata_file, "rb") as f:
        metadata = pickle.load(f)

    return metadata


def video_generator(path_to_video_file):
    cap = cv2.VideoCapture(path_to_video_file)
    while True:
        yield cap.read()


def render_video_with_metadata(video, metadata):
    bboxes = metadata["results"]
    for (ret, frame), bbox in zip(video, bboxes):

        if not ret:
            break

        frame = render_bboxes_on_frame(frame, bboxes)
        cv2.imshow(f"{metadata['trace_name']}", frame)
        cv2.waitKey(30)


def render_bboxes_on_frame(img, annotations):

    return img


if __name__ == "__main__":
    main()
