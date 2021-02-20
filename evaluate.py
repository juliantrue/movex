import sys, os
import time
import queue
import av
import cv2
import pickle
from fig import Config as C
from tqdm import tqdm

from deepsort import DeepSortTracker
from accumulator import Accumulator
from movex import MOTClient, extract_mvs, apply_mvs, apply_queue_of_mvs


def run_mot16_eval(path_to_mot_dir, path_to_results_dir):
    subset = [
        "MOT16-02",
        "MOT16-09",
        "MOT16-11",
        "MOT16-04",
        "MOT16-05",
        "MOT16-10",
        "MOT16-13",
    ]
    for mot_trace_dir in os.listdir(path_to_mot_dir):
        if mot_trace_dir in subset:
            path_to_trace_dir = os.path.join(path_to_mot_dir, mot_trace_dir)
            run_data = run_mot16_trace(path_to_trace_dir, path_to_results_dir)


def run_mot16_trace(path_to_trace_dir, path_to_results_dir):
    """Runs tracking with MoVe Extrapolation on the MOT16 trace at `path_to_trace_dir`
    and outputs the results to the `path_to_results_dir`"""
    trace_name = os.path.split(path_to_trace_dir)[1]
    video_source = read_video_source_and_set_read_settings(
        os.path.join(path_to_trace_dir, trace_name + ".mp4")
    )

    client_config = {
        "path_to_detections": os.path.join(path_to_trace_dir, trace_name + ".npy"),
        "latency": C.move_config_dnn_simulator_latency_in_ms,
        "video_fps": video_source.streams.video[0].average_rate,
    }
    detection_source = MOTClient(client_config)

    print(f"Running tracking on {trace_name}.")
    run_data = run_tracking(video_source, detection_source)
    run_data["trace_name"] = trace_name

    path_to_results_file = os.path.join(path_to_results_dir, trace_name + ".txt")
    path_to_run_data_file = os.path.join(path_to_results_dir, trace_name + ".metadata")
    write_results_to_results_dir(run_data["results"], path_to_results_file)
    write_run_data_to_dir(run_data, path_to_run_data_file)
    return run_data


def read_video_source_and_set_read_settings(path_to_video):
    video_source = av.open(path_to_video)
    video_source.streams.video[0].export_mvs = True
    return video_source


def run_tracking(video_source, detection_source):
    """Runs tracking with MoVe Extrapolation based on a av.container video source and
    a detection source that implements `poll` behaviour."""

    mv_filter_method = "alpha_trim"
    tracker = DeepSortTracker(
        C.move_config_deep_sort_nn_budget,
        C.move_config_deep_sort_max_cosine_distance,
        C.move_config_deep_sort_nms_max_overlap,
        C.move_config_deep_sort_min_confidence,
        C.move_config_deep_sort_max_iou_distance,
        C.move_config_deep_sort_n_init,
    )
    acc = Accumulator()
    acc.register(extract_mvs)
    acc.register(apply_queue_of_mvs)
    acc.register(apply_mvs)
    run_data = {
        "results": [],
        "statistics": {
            "avg_loop_time": [],
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
            detections = detection_source.await(i)
            curr_bboxes = detections[0]

        else:
            curr_mvs = extract_mvs(frame)
            mvs_buffer.put(curr_mvs)

            detections = detection_source.poll(i)
            if first_frame and detections is None:
                print("Skipping loop iteration until first detections come in.")
                continue

            elif not first_frame and detections is None:
                curr_bboxes, mvs = apply_mvs(curr_bboxes, curr_mvs, mv_filter_method)

            else:
                first_frame = False
                bboxes = detections[0]
                bboxes = apply_queue_of_mvs(mvs_buffer, bboxes, mv_filter_method)
                curr_bboxes = bboxes.copy()

        curr_scores = [1] * len(curr_bboxes)
        tracked_bboxes, track_ids = tracker.track(
            img, curr_bboxes, curr_scores, tlbr=False
        )

        loop_now = time.perf_counter()
        run_data["statistics"]["avg_loop_time"].append(loop_now - loop_last)

        for j in range(len(tracked_bboxes)):
            run_data["results"].append(
                [
                    i,
                    track_ids[j],
                    tracked_bboxes[j][0],
                    tracked_bboxes[j][1],
                    tracked_bboxes[j][2],
                    tracked_bboxes[j][3],
                ]
            )

    run_data["statistics"]["debugging"].show()
    run_data["statistics"] = post_process_statistics(run_data["statistics"])
    return run_data


def unpack_frame_to_img(frame):
    img = frame.to_ndarray(format="rgb24")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def post_process_statistics(statistics):
    avg_loop_time = sum(statistics["avg_loop_time"]) / len(statistics["avg_loop_time"])
    statistics["avg_loop_time"] = avg_loop_time
    try:
        statistics["debugging"] = statistics["debugging"].as_json()
    except Exception as e:
        print(e)
        print("Skipping debugging metric computation.")

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


def display_eval_metadata():
    folder = sys.argv[1]
    print(f"Displaying run statistic for: {folder}")
    avg_loop_time = compute_eval_metadata(folder)
    print(f"Average Loop Time: {avg_loop_time*1000}ms")


def compute_eval_metadata(path_to_results_dir):
    avg_avg_loop_time = []
    for result_file in os.listdir(path_to_results_dir):
        if result_file.split(".")[-1] == "metadata":
            path_to_metadata_file = os.path.join(path_to_results_dir, result_file)
            run_data = parse_metadata_file(path_to_metadata_file)
            avg_avg_loop_time.append(run_data["statistics"]["avg_loop_time"])

    return sum(avg_avg_loop_time) / len(avg_avg_loop_time)


def parse_metadata_file(path_to_metadata_file):
    with open(path_to_metadata_file, "rb") as f:
        run_data = pickle.load(f)

    return run_data


if __name__ == "__main__":
    display_eval_metadata()
