import time
import cv2
import numpy as np
from numpy.lib import recfunctions as rfn
from accumulator import collect_run_time

from fig import Config as C
from .decorators import rolling_frame


@collect_run_time
def apply_queue_of_mvs(mvs_buffer, bboxes, method):
    for _ in range(mvs_buffer.qsize()):
        mvs = mvs_buffer.get()
        bboxes = apply_mvs(bboxes, mvs, method)

    return bboxes


@collect_run_time
def apply_mvs(bboxes, mvs, method):
    """perturbs bboxes according to update rule

    bboxes: [[x, y, width, height]]
    """
    out_bboxes = []
    out_mvs = []
    if len(mvs) == 0:
        return out_bboxes

    elif C.move_config_ablation_skip_perturbation:
        return bboxes

    for bbox in bboxes:
        # Get all MVs contained in the bbox
        mvs_in_bbox = crop_mvs_to_bbox(mvs, bbox)
        num_mvs, _ = mvs_in_bbox.shape
        if num_mvs == 0:
            # If there are no mvs in the box?
            motion_xy_by_scale = np.array([0, 0])

        else:
            motion_xy_by_scale = filter_bbox_mvs(mvs_in_bbox, method)

        # Apply the motion vector to the bbox
        motion_x_by_scale, motion_y_by_scale = motion_xy_by_scale

        # src_x = dst_x + motion_x / motion_scale
        # Solve for dst using above
        # See: https://github.com/FFmpeg/FFmpeg/blob/a0ac49e38ee1d1011c394d7be67d0f08b2281526/libavutil/motion_vector.h
        bbox_to_append = [
            bbox[0] - motion_x_by_scale,  # x
            bbox[1] - motion_y_by_scale,  # y
            bbox[2],  # w
            bbox[3],  # h
        ]
        out_bboxes.append(bbox_to_append)
        out_mvs.append(motion_xy_by_scale)
    return out_bboxes


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


def filter_bbox_mvs(mvs_in_bbox, method):
    num_mvs, _ = mvs_in_bbox.shape

    # Compute the median x motion and median y motion
    # x and y * motion_scale
    motion_scale = np.repeat(np.expand_dims(mvs_in_bbox[:, 6], axis=1), 2, axis=1)
    motion_xy_by_scale = mvs_in_bbox[:, 4:6] / motion_scale

    if method == "median":
        motion_xy_by_scale = np.median(motion_xy_by_scale, axis=0)

    elif method == "mean":
        motion_xy_by_scale = np.mean(motion_xy_by_scale, axis=0)

    elif method == "alpha_trim":
        magnitudes = np.linalg.norm(motion_xy_by_scale, axis=1)
        magnitudes_argsort_idxs = np.argsort(magnitudes, axis=0)
        motion_xy_by_scale = motion_xy_by_scale[magnitudes_argsort_idxs, :]
        trim = int((num_mvs * C.move_config_alpha_trim_alpha) // 2)
        motion_xy_by_scale = motion_xy_by_scale[trim:-trim, :]
        if len(motion_xy_by_scale) == 0:
            motion_xy_by_scale = np.array([0, 0])

        else:
            motion_xy_by_scale = np.median(motion_xy_by_scale, axis=0)

    else:
        print("Undefined method, using median")
        motion_xy_by_scale = np.median(motion_xy_by_scale, axis=0)

    return motion_xy_by_scale


@collect_run_time
def extract_mvs(frame):
    method = "flownet"
    mvs = np.array([])
    if method == "h264":
        mvs = h264_mvs(frame)

    elif method == "rlof":
        mvs = rlof(frame)

    elif method == "flownet":
        mvs = flownet(frame, C.current_trace)

    return mvs


def h264_mvs(frame):
    mvs = list(frame.side_data)
    if len(mvs) > 0:
        mvs = mvs[0]

        # https://github.com/FFmpeg/FFmpeg/blob/a0ac49e38ee1d1011c394d7be67d0f08b2281526/libavutil/motion_vector.h
        # Motion Vector:
        # src_x = dst_x + motion_x / motion_scale
        # src_y = dst_y + motion_x / motion_scale
        mvs = mvs.to_ndarray()
        mvs = rfn.structured_to_unstructured(mvs)

        # returns a Nx7 array where there are N motion vectors defined by:
        # [src_x, src_y, dst_x, dst_y, motion_x, motion_y, motion_scale]
        mvs = mvs[:, [3, 4, 5, 6, 8, 9, 10]]

    else:
        mvs = np.array([])

    return mvs


@rolling_frame
def rlof(curr_frame, last_frame):
    mvs = np.array([])
    if last_frame is not None:
        flow = cv2.optflow.calcOpticalFlowDenseRLOF(
            last_frame, curr_frame, None, gridStep=(16, 16)
        )
        flow_x = flow[:, :, 0].reshape(-1)
        flow_y = flow[:, :, 1].reshape(-1)

        w, h, _ = flow.shape
        src_x = np.arange(0, w)
        src_y = np.arange(0, h)

        arr = np.array(np.meshgrid(src_x, src_y)).T.reshape(-1, 2)
        src_x, src_y = arr[:, 0], arr[:, 1]
        dst_x, dst_y = src_x - flow_y, src_y - flow_y

        src_x, src_y = np.expand_dims(src_x, axis=1), np.expand_dims(src_y, axis=1)
        dst_x, dst_y = np.expand_dims(dst_x, axis=1), np.expand_dims(dst_y, axis=1)
        flow_x, flow_y = np.expand_dims(flow_x, axis=1), np.expand_dims(flow_y, axis=1)

        mvs = np.concatenate(
            [src_x, src_y, dst_x, dst_y, flow_x, flow_y, np.ones(flow_x.shape)], axis=1
        )

    return mvs


def flownet(curr_frame, mot_trace):

    last = time.perf_counter()
    # I know this isn't pretty, but if we are using flownet, we need this precomputed for simplicity
    if not hasattr(C, "current_frame"):
        C.current_frame = 0
        frame_idx = str(C.current_frame).zfill(7)

    else:
        C.current_frame += 1
        frame_idx = str(C.current_frame).zfill(7)

    mvs = np.array([])
    filename = f"/flownet_data/{mot_trace}/outputs/{frame_idx}-flow.flo"
    f = open(filename, "rb")

    et_str = f.read(13)
    elapsed_time_in_s = float(et_str)

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()
    flow = np.fromfile(f, np.float32, width * height).reshape((height, width))
    flow = flow.astype(np.float32)

    flow_x = flow[:, 0].reshape(-1)
    flow_y = flow[:, 1].reshape(-1)

    last = time.perf_counter()
    src_x = np.arange(0, height)
    src_y = np.arange(0, height)

    dst_x, dst_y = src_x - flow_y, src_y - flow_y

    last = time.perf_counter()
    src_x, src_y = np.expand_dims(src_x, axis=1), np.expand_dims(src_y, axis=1)
    dst_x, dst_y = np.expand_dims(dst_x, axis=1), np.expand_dims(dst_y, axis=1)
    flow_x, flow_y = np.expand_dims(flow_x, axis=1), np.expand_dims(flow_y, axis=1)

    mvs = np.concatenate(
        [src_x, src_y, dst_x, dst_y, flow_x, flow_y, np.ones(flow_x.shape)], axis=1
    )

    now = time.perf_counter()
    elapsed_time_in_s_actual = now - last
    delta_t = elapsed_time_in_s - elapsed_time_in_s_actual
    if not delta_t < 0:
        time.sleep(elapsed_time_in_s - elapsed_time_in_s_actual)

    return mvs
