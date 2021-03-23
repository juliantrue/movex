import sys, os
import pickle
from fig import Config as C


def display_eval_metadata():
    folder = sys.argv[1]
    print(C.show())
    print(f"Displaying run statistic for: {folder}")
    avg_loop_time, cum_loop_time = compute_eval_metadata(folder)
    print(f"Average Loop Time: {avg_loop_time*1000}ms")
    print(f"Cumulative Loop Time: {cum_loop_time*1000}ms")


def compute_eval_metadata(path_to_results_dir):
    avg_avg_loop_time = []
    cumulative_time = 0
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
            run_data = parse_metadata_file(path_to_metadata_file)
            avg_avg_loop_time.append(run_data["statistics"]["avg_loop_time"])

    return sum(avg_avg_loop_time) / len(avg_avg_loop_time), cumulative_time


def parse_metadata_file(path_to_metadata_file):
    with open(path_to_metadata_file, "rb") as f:
        run_data = pickle.load(f)

    return run_data


if __name__ == "__main__":
    display_eval_metadata()
