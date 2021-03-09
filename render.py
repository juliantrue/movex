import os
import pickle
import cv2
import numpy as np


def main():
    path_to_result_metadata_file = "results/test_run_ignore_mvs/MOT16-04.metadata"
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
    results = metadata["results"]
    results = np.array(results)
    bboxes_per_frame = bbox_generator(results)

    for (ret, frame), bboxes in zip(video, bboxes_per_frame):

        if not ret:
            break

        frame = render_bboxes_on_frame(frame, bboxes)
        cv2.imshow(f"{metadata['trace_name']}", frame)
        cv2.waitKey(30)


def bbox_generator(results):
    bboxes_xyah = results[:, 2:]
    bboxes_tlbr = bboxes_xyah.copy()
    bboxes_tlbr[:, 2] = bboxes_xyah[:, 0] + bboxes_xyah[:, 2]
    bboxes_tlbr[:, 3] = bboxes_xyah[:, 1] + bboxes_xyah[:, 3]

    frame_idxs_with_duplicates = results[:, 0]
    frames_idxs = np.unique(frame_idxs_with_duplicates)
    for i in frames_idxs:
        bbox_idxs = np.where(frame_idxs_with_duplicates == i)
        yield np.squeeze(bboxes_tlbr[bbox_idxs, :])


def render_bboxes_on_frame(img, annotations):
    for bbox in annotations:
        bbox = list(map(lambda x: int(x), bbox))
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))

    return img


if __name__ == "__main__":
    main()