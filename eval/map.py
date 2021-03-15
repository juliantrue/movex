import sys
import pandas as pd
import numpy as np
from map_boxes import mean_average_precision_for_boxes


def map_eval(path_to_detections_file, path_to_ground_truth_file):
    ann = pd.read_csv(path_to_ground_truth_file)
    det = pd.read_csv(path_to_detections_file)
    ann = ann[["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"]].values
    det = det[["ImageID", "LabelName", "Conf", "XMin", "XMax", "YMin", "YMax"]].values
    mean_ap, average_precisions = mean_average_precision_for_boxes(ann, det)


if __name__ == "":
    argvs = sys.argv
    if len(argvs) < 3:
        print("Missing args")
        print("USAGE: map.py path_to_detections_file path_to_ground_truth_file")
        sys.exit(0)

    map_eval(argvs[1], argvs[2])
