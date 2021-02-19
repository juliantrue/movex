import time
import numpy as np
from numpy.lib import recfunctions as rfn
from accumulator import collect_run_time

from fig import Config as C
from movexo3 import mvs_to_list


@collect_run_time
def apply_queue_of_mvs(mvs_buffer, bboxes):
    for _ in range(mvs_buffer.qsize()):
        mvs = mvs_buffer.get()
        bboxes, mvs = apply_mvs(bboxes, mvs)

    return bboxes


@collect_run_time
def apply_mvs(bboxes, mvs, method="median"):
    """perturbs bboxes according to update rule

    bboxes: [[x, y, width, height]]
    """
    out_bboxes = []
    out_mvs = []
    if len(mvs) == 0:
        return out_bboxes, out_mvs

    for bbox in bboxes:
        # Get all MVs contained in the bbox
        mvs_in_bbox = crop_mvs_to_bbox(mvs, bbox)

        if len(mvs_in_bbox) == 0:
            # If there are no mvs in the box?
            motion_xy_by_scale = np.array([0, 0])

        else:
            # Compute the median x motion and median y motion
            # x and y * motion_scale
            motion_scale = np.repeat(
                np.expand_dims(mvs_in_bbox[:, 6], axis=1), 2, axis=1
            )
            motion_xy_by_scale = mvs_in_bbox[:, 4:6] / motion_scale

            if method == "median":
                motion_xy_by_scale = np.median(motion_xy_by_scale, axis=0)

            elif method == "mean":
                motion_xy_by_scale = np.mean(motion_xy_by_scale, axis=0)

            else:
                print("Undefined method, using median")
                motion_xy_by_scale = np.median(motion_xy_by_scale, axis=0)

        # Apply the motion vector to the bbox
        motion_x_by_scale, motion_y_by_scale = motion_xy_by_scale

        # src_x = dst_x + motion_x / motion_scale
        # Solve for dst using above
        # See: https://github.com/FFmpeg/FFmpeg/blob/a0ac49e38ee1d1011c394d7be67d0f08b2281526/libavutil/motion_vector.h
        out_bboxes.append(
            (
                [
                    bbox[0] - motion_x_by_scale,  # x
                    bbox[1] - motion_y_by_scale,  # y
                    bbox[2],  # w
                    bbox[3],  # h
                ]
            )
        )
        out_mvs.append(motion_xy_by_scale)
    return out_bboxes, out_mvs


def crop_mvs_to_bbox(mvs, bbox):
    """
    mvs: [[src_x, src_y, dst_x, dst_y, motion_x, motion_y, motion_scale] * N]
    bboxes: [[x, y, width, height] * N]
    """
    where_src_x = np.logical_and(bbox[0] <= mvs[:, 0], mvs[:, 0] <= bbox[0] + bbox[2])
    where_src_y = np.logical_and(bbox[1] <= mvs[:, 1], mvs[:, 1] <= bbox[1] + bbox[3])
    where__x_and_y = np.logical_and(where_src_x, where_src_y)
    mvs_in_bbox = mvs[where__x_and_y]

    return mvs_in_bbox


@collect_run_time
def extract_mvs(frame):
    curr_mvs = list(frame.side_data)
    if len(curr_mvs) > 0:
        curr_mvs = curr_mvs[0]

        # https://github.com/FFmpeg/FFmpeg/blob/a0ac49e38ee1d1011c394d7be67d0f08b2281526/libavutil/motion_vector.h
        # Motion Vector:
        # src_x = dst_x + motion_x / motion_scale
        # src_y = dst_y + motion_x / motion_scale
        curr_mvs = curr_mvs.to_ndarray()
        curr_mvs = rfn.structured_to_unstructured(curr_mvs)

        # returns a Nx7 array where there are N motion vectors defined by:
        # [src_x, src_y, dst_x, dst_y, motion_x, motion_y, motion_scale]
        curr_mvs = curr_mvs[:, [3, 4, 5, 6, 8, 9, 10]]

    else:
        curr_mvs = np.array([])

    return curr_mvs
