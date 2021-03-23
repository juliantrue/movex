import sys, os
import numpy as np
import json


def main(*args):
    source_file = args[0]
    target_folder = args[1]
    if len(args) > 2:
        print("Too many args")
        print(
            "USAGE: python3 mot_to_absolute path_to_source_file path_to_target_dir --include_confidence"
        )
        sys.exit(0)

    # Convert file path to abs detections
    data = convert(source_file)

    # Create a folder for results
    updated_target_folder = make_target_directory(target_folder)

    # Write to target folder
    write_to_target_directory(data, updated_target_folder)


def make_target_directory(path_to_target_dir):
    curr_iteration = 0
    makedir_path = path_to_target_dir
    while os.path.exists(makedir_path):
        curr_iteration += 1
        makedir_path = path_to_target_dir + "{}"
        makedir_path = makedir_path.format(str(curr_iteration))

    os.makedirs(makedir_path)

    return makedir_path


def convert(path_to_source_file):
    # Parse files to numpy arrays
    data = np.loadtxt(path_to_source_file, delimiter=",")
    rows, cols = data.shape
    include_confidence = cols == 10

    # Split by frames
    frames = np.unique(data[:, 0])

    out = dict()
    for frame in frames:
        idxs = np.where(data[:, 0] == frame)
        frame_data = data[idxs]

        bboxes = extract_bboxes(frame_data)

        labels = np.expand_dims(np.repeat(["person"], len(bboxes)), axis=1)
        if include_confidence:
            data_out = np.concatenate(
                [labels, np.ones([len(bboxes), 1], dtype=np.uint64), bboxes],
                axis=1,
            )

        else:
            data_out = np.concatenate([labels, bboxes], axis=1)

        out[frame] = data_out

    return out


def extract_bboxes(frame_data):

    idxs = np.where(frame_data[:, 6] != 0)
    frame_data = frame_data[idxs]
    bboxes = frame_data[:, 2:6]  # N x (left, top, width, height)
    # bboxes[:, 2] += bboxes[:, 0]
    # bboxes[:, 3] += bboxes[:, 1]  # N x (left, top, right, bottom)
    bboxes = bboxes.astype(np.int64)

    return bboxes


def write_to_target_directory(data, path_to_target_dir):

    fname = os.path.split(path_to_target_dir)[1].split(".")[0]
    out = {
        "videos": [{"id": 0, "file_name": fname, "width": 1080, "height": 1920}],
        "annotations": [],
        "categories": [{"id": 1, "name": "person"}],
    }

    annotation_template = {
        "id": int,
        "video_id": int,
        "category_id": 1,
        "track": [{"frame": int, "bbox": [x, y, width, height], "confidence": 1}],
    }

    keys = list(data.keys())
    first_key = keys[0]

    rows, cols = data[first_key].shape
    if cols == 6:
        # Its detections
        pass

    else:
        # Its ground truths
        for k in keys:
            annotation = annotation_template.copy()
            frame_data = data[k]

            out["annotations"].append(annotation)


if __name__ == "__main__":
    main(*sys.argv[1:])
