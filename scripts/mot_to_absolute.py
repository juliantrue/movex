import sys, os
import numpy as np


def main(*args):
    if len(args) > 3:
        print("Too many args")
        print(
            "USAGE: python3 mot_to_absolute.py [path_to_source_file] [path_to_target_dir] OPTIONAL [--include_confidence]"
        )
        sys.exit(0)

    elif len(args) < 2:
        print("Too few args")
        print(
            "USAGE: python3 mot_to_absolute.py [path_to_source_file] [path_to_target_dir] OPTIONAL [--include_confidence]"
        )
        sys.exit(0)
    source_file = args[0]
    target_folder = args[1]

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
    first_key = 1
    checked = False
    for k in data:
        if not k == first_key and not checked:
            checked = True
            delta = int(k - first_key)
            for i in range(1, delta + 1):
                file_name = str(i).zfill(6)
                target_file_path = os.path.join(path_to_target_dir, file_name + ".txt")
                f = open(target_file_path, "w")
                f.close()

        file_name = str(int(k)).zfill(6)
        target_file_path = os.path.join(path_to_target_dir, file_name + ".txt")

        data_at_k = data[k]
        rows, cols = data_at_k.shape

        if cols == 6:
            fmt = "%s %s %s %s %s %s"

        else:
            fmt = "%s %s %s %s %s"

        np.savetxt(target_file_path, data_at_k, fmt=fmt)


if __name__ == "__main__":
    main(*sys.argv[1:])
