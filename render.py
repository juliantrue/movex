import os
import pickle
import cv2
import numpy as np


def main():
    mot_trace_name = "MOT20-03"
    path_to_results_dir = "results/baseline/sandbox/latency300"
    path_to_data_dir = "data"

    path_to_result_metadata_file = os.path.join(
        path_to_results_dir, f"{mot_trace_name}.metadata"
    )
    metadata = parse_result_metadata_file(path_to_result_metadata_file)
    path_to_video_file = os.path.join(
        path_to_data_dir, mot_trace_name, mot_trace_name + ".mp4"
    )

    video = video_generator(path_to_video_file)
    render_video_with_metadata(video, metadata)


def parse_result_metadata_file(path_to_result_metadata_file):
    with open(path_to_result_metadata_file, "rb") as f:
        metadata = pickle.load(f)

    return metadata


def render_video_with_metadata(video, metadata):
    results = metadata["results"]
    results = np.array(results)
    bboxes_per_frame = bbox_generator(results)

    for (ret, frame), bboxes in zip(video, bboxes_per_frame):

        if not ret:
            break

        frame = render_bboxes_on_frame(frame, bboxes)
        cv2.imshow(f"{metadata['trace_name']}", frame)
        cv2.waitKey(25)


def video_generator(path_to_video_file):
    cap = cv2.VideoCapture(path_to_video_file)
    while True:
        yield cap.read()


def bbox_generator(results):
    bboxes_xyah = results[:, 2:]
    bboxes_tlbr = bboxes_xyah.copy()
    bboxes_tlbr[:, 2] = bboxes_xyah[:, 0] + bboxes_xyah[:, 2]
    bboxes_tlbr[:, 3] = bboxes_xyah[:, 1] + bboxes_xyah[:, 3]

    frame_idxs_with_duplicates = results[:, 0]
    frames_idxs = np.sort(np.unique(frame_idxs_with_duplicates))
    first_idx = frames_idxs[0]
    prelude = int(first_idx)

    for i in range(int(frames_idxs[-1])):
        bbox_idxs = np.where(frame_idxs_with_duplicates == i)
        if len(bbox_idxs[0]) == 0:
            yield []

        else:
            yield np.squeeze(bboxes_tlbr[bbox_idxs, :])


def render_bboxes_on_frame(img, annotations):
    for bbox in annotations:
        bbox = list(map(lambda x: int(x), bbox))
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))

    return img


if __name__ == "__main__":
    main()
