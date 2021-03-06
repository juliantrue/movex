import sys, os
import time
import queue
import av
import cv2
import pickle
import numpy as np
from fig import Config as C
from tqdm import tqdm


from accumulator import Accumulator
from movex import (
    MOTClient,
    SingleThreadedClientSim,
    extract_mvs,
    apply_mvs,
    apply_queue_of_mvs,
)


def run_mot_eval(path_to_mot_dir, path_to_results_dir):
    subset = C.app_mot_eval_subset
    for mot_trace_dir in os.listdir(path_to_mot_dir):
        if mot_trace_dir in subset:
            path_to_trace_dir = os.path.join(path_to_mot_dir, mot_trace_dir)
            run_data = run_mot_trace(path_to_trace_dir, path_to_results_dir)


def run_mot_trace(path_to_trace_dir, path_to_results_dir):
    """Runs tracking with MoVe Extrapolation on the MOT16 trace at `path_to_trace_dir`
    and outputs the results to the `path_to_results_dir`"""
    trace_name = os.path.split(path_to_trace_dir)[1]
    C.current_trace = trace_name
    encoding = "" if C.app_mot_eval_encoding == "" else "-" + C.app_mot_eval_encoding
    video_source = read_video_source_and_set_read_settings(
        os.path.join(path_to_trace_dir, trace_name + encoding + ".mp4")
    )

    network = C.app_mot_eval_detections
    client_config = {
        "path_to_detections": os.path.join(
            path_to_trace_dir,
            trace_name + f"{'-' + network if len(network) > 0 else ''}" + ".npy",
        ),
        "latency": C.move_config_dnn_simulator_latency_in_ms,
        "video_fps": video_source.streams.video[0].average_rate,
    }

    if C.move_config_ablation_single_threaded:
        detection_source = SingleThreadedClientSim(client_config)

    else:
        detection_source = MOTClient(client_config)

    print(f"Running tracking on {trace_name}.")
    run_data = run_detection(video_source, detection_source)
    run_data["trace_name"] = trace_name

    if run_data["statistics"] is not None:
        path_to_run_data_file = os.path.join(
            path_to_results_dir, trace_name + ".metadata"
        )
        write_run_data_to_dir(run_data, path_to_run_data_file)

    path_to_results_file = os.path.join(path_to_results_dir, trace_name + ".txt")
    write_results_to_results_dir(run_data["results"], path_to_results_file)
    return run_data


def read_video_source_and_set_read_settings(path_to_video):
    video_source = av.open(path_to_video)
    video_source.streams.video[0].export_mvs = True
    return video_source


def run_detection(video_source, detection_source):
    """Runs detection with MoVe Extrapolation based on a av.container video source and
    a detection source that implements `poll` behaviour."""

    mv_filter_method = C.move_config_aggregation_method
    acc = Accumulator()
    acc.register(extract_mvs)
    acc.register(apply_queue_of_mvs)
    acc.register(apply_mvs)
    run_data = {
        "results": [],
        "statistics": {
            "loop_time_samples": [],
            "debugging": acc,
        },
    }

    first_frame = True
    curr_mvs = None
    curr_bboxes = None
    mvs_buffer = queue.Queue()
    for i, frame in tqdm(enumerate(video_source.decode(video=0))):
        img = unpack_frame_to_img(frame)

        loop_last = time.perf_counter()

        if C.move_config_bypass:
            detections = detection_source.wait(i)
            curr_bboxes = detections[0]

        else:
            curr_mvs = extract_mvs(frame)
            mvs_buffer.put(curr_mvs)

            detections = detection_source.poll(i)
            if first_frame and detections is None:
                print("Skipping loop iteration until first detections come in.")
                continue

            elif not first_frame and detections is None:
                curr_bboxes = apply_mvs(curr_bboxes, curr_mvs, mv_filter_method)

            else:
                first_frame = False
                bboxes = detections[0]
                bboxes = apply_queue_of_mvs(mvs_buffer, bboxes, mv_filter_method)
                curr_bboxes = bboxes.copy()

        loop_now = time.perf_counter()
        run_data["statistics"]["loop_time_samples"].append(loop_now - loop_last)

        h, w, _ = img.shape
        track_ids = [-1] * len(curr_bboxes)
        for j in range(len(curr_bboxes)):
            bbox = clamp_bbox_to_frame(curr_bboxes[j], (h, w))
            run_data["results"].append(
                [
                    i,
                    track_ids[j],
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                ]
            )

    try:
        run_data["statistics"] = post_process_statistics(run_data["statistics"])

    except Exception as e:
        print("ERROR: ", e)
        run_data["statistics"] = None

    return run_data


def unpack_frame_to_img(frame):
    img = frame.to_ndarray(format="rgb24")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def clamp_bbox_to_frame(bbox, img_shape):
    h, w = img_shape
    if bbox[0] < 0:
        bbox[0] = 0

    elif bbox[0] > w:
        bbox[0] = w

    if bbox[1] < 0:
        bbox[1] = 0

    elif bbox[1] > h:
        bbox[1] = h

    return bbox


def post_process_statistics(statistics):
    avg_loop_time = sum(statistics["loop_time_samples"]) / len(
        statistics["loop_time_samples"]
    )
    statistics["avg_loop_time"] = avg_loop_time

    return statistics


def write_results_to_results_dir(results, path_to_results_file):

    results_dir = os.path.split(path_to_results_file)[0]
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(path_to_results_file, "w") as f:
        for row in results:
            print(
                "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1"
                % (row[0], row[1], row[2], row[3], row[4], row[5]),
                file=f,
            )


def write_run_data_to_dir(run_data, path_to_results_file):
    results_dir = os.path.split(path_to_results_file)[0]
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(path_to_results_file, "wb") as f:
        pickle.dump(run_data, f)
