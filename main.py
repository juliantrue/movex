from fig import Config as C

from eval import run_mot_eval, run_mot_trace


def main():
    print("App configuration:")
    C.show()
    config_as_dict = C.as_dict()
    app_config = config_as_dict["app"]
    app = list(app_config.keys())[0]

    if len(app_config) > 1:
        raise Exception(
            f"Multiple apps specified: {list(app_config.keys())}. "
            + "Only one type of app should be specified."
        )

    elif app == "mot_eval":

        if isinstance(C.move_config_dnn_simulator_latency_in_ms, list):
            latencies = C.move_config_dnn_simulator_latency_in_ms
            base_dir = C.app_mot_eval_results_dir
            for latency in latencies:
                C.move_config_dnn_simulator_latency_in_ms = latency
                C.app_mot_eval_results_dir = base_dir + str(latency)

                print("\n\nRunning Evaluation With Latency: ", latency, "\n\n")
                run_mot_eval(C.app_mot_eval_path_to_mot_dir, C.app_mot_eval_results_dir)

        else:
            run_mot_eval(C.app_mot_eval_path_to_mot_dir, C.app_mot_eval_results_dir)

    elif app == "mot_trace":
        run_mot_trace(
            C.app_mot_trace_path_to_mot_trace_dir, C.app_mot_trace_results_dir
        )

    else:
        raise Exception(
            f"{app} is not in list of available applications: "
            + '["mot_eval", mot_trace]'
        )


if __name__ == "__main__":
    main()
