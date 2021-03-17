import sys
import os
import pandas as pd
import numpy as np
from map_boxes import mean_average_precision_for_boxes


def map_eval_folder(path_to_mot_folder, path_to_detections_folder):
    gt_train_dir_path = os.path.join(path_to_mot_folder, "train")
    gt_folders = os.listdir(gt_train_dir_path)
    for gt_folder_path in gt_folders:
        gt_path = os.path.join(gt_train_dir_path, gt_folder_path, "gt", "gt.txt")
        seqini_path = os.path.join(gt_train_dir_path, gt_folder_path, "seqinfo.ini")
        gt_mot_trace_name = gt_folder_path.split(".")[0]

        det_files = os.listdir(os.path.join(path_to_detections_folder, "train"))
        for det_file in det_files:
            det_path = os.path.join("train", det_file, "det/det.txt")
            det_path = os.path.join(path_to_detections_folder, det_path)
            # if len(fn_split) < 2:
            #    continue

            # det_mot_trace_name, extension = fn_split[0], fn_split[1]
            det_mot_trace_name = det_file
            # if not extension == "txt":
            #    continue

            if gt_mot_trace_name == det_mot_trace_name:
                map_eval(gt_path, det_path, seqini_path)


def map_eval(path_to_ground_truth_file, path_to_detections_file, seqini_path):
    print("Running map eval on: ", path_to_ground_truth_file, path_to_detections_file)
    dets = np.loadtxt(path_to_detections_file, delimiter=",")
    dets = dets[:, :6]
    dets[:, 0] = dets[:, 0].astype(np.str)
    dets[:, 1] = np.ones(len(dets))
    dets[:, 2:] = tlwh_to_xmym(dets[:, 2:])
    dets[:, 2:] = normalize_bboxes(seqini_path, dets[:, 2:])
    left = dets[:, :2]
    right = dets[:, 2:]
    dets = np.concatenate([left, np.ones([len(dets), 1]), right], axis=1)

    print(
        pd.DataFrame(
            dets,
            columns=["ImageID", "LabelName", "Conf", "XMin", "XMax", "YMin", "YMax"],
        )
    )

    gts = np.loadtxt(path_to_ground_truth_file, delimiter=",")
    gts = gts[:, :6]
    dets[:, 0] = gts[:, 0].astype(np.str)
    gts[:, 1] = np.ones(len(gts))
    gts[:, 2:] = tlwh_to_xmym(gts[:, 2:])
    gts[:, 2:] = normalize_bboxes(seqini_path, gts[:, 2:])
    print(
        pd.DataFrame(
            gts,
            columns=["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"],
        )
    )

    mean_ap, average_precisions = mean_average_precision_for_boxes(
        gts, dets, iou_threshold=0.01, verbose=True
    )


def tlwh_to_xmym(tlwh):
    """N x 4 nd array where 0: x_min, 1: y_min, 2: width 3: height"""
    x_max = tlwh[:, 0] + tlwh[:, 2]
    y_max = tlwh[:, 1] + tlwh[:, 3]
    xmym = tlwh.copy()
    xmym[:, 1] = x_max
    xmym[:, 2] = tlwh[:, 1]
    xmym[:, 3] = y_max

    return xmym


def normalize_bboxes(path_to_seqini, bboxes):
    metadata = extract_metadata(path_to_seqini)
    w = int(metadata["imWidth"])
    h = int(metadata["imHeight"])

    bboxes[:, 0] = bboxes[:, 0] / w
    bboxes[:, 2] = bboxes[:, 2] / w
    bboxes[:, 1] = bboxes[:, 0] / h
    bboxes[:, 3] = bboxes[:, 2] / h
    return bboxes


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


if __name__ == "__main__":
    argvs = sys.argv
    if len(argvs) < 3:
        print("Missing args")
        print("USAGE: map.py  path_to_ground_truth_file path_to_detections_file")
        sys.exit(0)

    map_eval_folder(argvs[1], argvs[2])
