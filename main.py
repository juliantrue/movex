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
