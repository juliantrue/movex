import sys, os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from fig import Config as C


def display_eval_metadata():
    folder = sys.argv[1]
    print(C.show())
    print(f"Displaying run statistic for: {folder}")
    avg_loop_time, cum_loop_time = compute_eval_metadata(folder)
    print(f"Average Loop Time: {avg_loop_time*1000}ms")
    print(f"Cumulative Loop Time: {cum_loop_time*1000}ms")

    plot_loop_time_vs_frame(folder)


def compute_eval_metadata(path_to_results_dir):
    avg_avg_loop_time = []
    cumulative_time = 0
    for path_to_metadata_file in metadata_file_paths_in_folder(path_to_results_dir):
            run_data = parse_metadata_file(path_to_metadata_file)
            avg_avg_loop_time.append(run_data["statistics"]["avg_loop_time"])
            cumulative_time += compute_total_run_time(
                run_data["statistics"]["loop_time_samples"]
            )

    return sum(avg_avg_loop_time) / len(avg_avg_loop_time), cumulative_time

def metadata_file_paths_in_folder(path_to_results_dir):
    metadata_paths = []
    for result_file in os.listdir(path_to_results_dir):
        result_file_split = result_file.split(".")
        path_to_potential_metadata_file = os.path.join(path_to_results_dir, result_file)
        if os.path.isdir(path_to_potential_metadata_file):
            continue

        elif (
            result_file_split[1] == "metadata"
            and result_file_split[0] in C.app_mot_eval_subset
        ):
            path_to_metadata_file = path_to_potential_metadata_file
            yield path_to_metadata_file




def compute_total_run_time(accumulator):
    return sum(accumulator)

def plot_loop_time_vs_frame(path_to_results_dir):
    all_data = {}
    for path_to_metadata_file in metadata_file_paths_in_folder(path_to_results_dir):
        run_data = parse_metadata_file(path_to_metadata_file)
        all_data[run_data["trace_name"]] = {k: v for k, v in run_data.items() if not k == "trace_name" }

    for trace_name in C.app_mot_eval_subset:
        if trace_name == "MOT20-05":
            samples = all_data[trace_name]["statistics"]["loop_time_samples"]
            samples = np.array(samples) * 1000
            if len(samples) > 300:
                samples = samples[:300]

            plt.plot(samples)

    plt.savefig("test.png")

def parse_metadata_file(path_to_metadata_file):
    with open(path_to_metadata_file, "rb") as f:
        run_data = pickle.load(f)

    return run_data


if __name__ == "__main__":
    display_eval_metadata()
